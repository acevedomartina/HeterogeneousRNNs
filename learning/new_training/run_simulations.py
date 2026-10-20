#  -------------------- Run simulations --------------------

# Updated: 20.10.2026

# Changes since 15.09.2026:
#   1. Noise: now generated ONCE per time step in evolution() (Itot = II + v + sigman * N(0,1))
#      and reused in both RK2 stages. Before, dynamics() drew independent noise in each
#      stage (separately for LIF and QIF). dynamics() no longer takes rng and has no noise term.
#      Trajectories are therefore not bit-for-bit reproducible against the previous version,
#      even with the same seed.
#   2. Neuron initialization: initialize_neurons() now takes (N, nqif, rng, vreset, vt).
#      QIF phases ~ U(0, 2*pi); LIF voltages ~ U(vreset, vt).
#      Before, all neurons (LIF included) started with x ~ U(0, 2*pi).
#   3. External current: currents() samples Ibac = amplitude * (low_frac + (high_frac - low_frac) * U),
#      i.e. uniform in amplitude * [low_frac, high_frac] (configurable, can be asymmetric).
#      Before, it was symmetric: uniform in [-amplitude, +amplitude].
#      low_frac / high_frac are saved in experiment.json.
#   4. Configuration through environment variables (MASTER_SEED, SIM_SIGMAN, SIM_CURRENT_AMPLITUDE,
#      SIM_CURRENT_LOW_FRAC, SIM_CURRENT_HIGH_FRAC, SIM_VRESET, SIM_GSYN, SIM_TARGET_AMPLITUDE,
#      SIM_CANT_SEED, SIM_PARALLEL_SEED, SIM_NAME), so parameter sweeps can be launched
#      from an external script without editing this file.
#      sim_name (and thus SAVE_DIR) now encodes the swept parameters unless SIM_NAME is set.
#   5. Parallelization is done at the seed level (parallelize_seed = SIM_PARALLEL_SEED);
#      pqif and target conditions run serially.
#   6. Output directory for connectivity matrices is created inside each worker
#      (path_w_seed.parent.mkdir) before saving, to be safe with parallel writes.
#   7. Core functions (dynamics, detect, evolution, initialize_neurons, initialize_connectivity_matrix,
#      initialize_training, currents) moved to snn_core.py, shared with the averaged-connectivity test
#      script. They no longer rely on module-level globals: N, sigman, alpha and the current
#      parameters (amplitude, low/high frac, itstim) are passed explicitly.


# Full code for running simulations where networks train to produce either oscillations or sequences at varying parameter sweeps (is set below).

# Options to set before running:
# Experiment
#    - MASTER_SEED: reproducibility seed for the complete experiment
#    - target_dynamics: choose "oscillations" or "sequences"
# Network and training
#    - N: number of neurons
#    - pqif_values: fractions of QIF neurons to test
#    - configs: threshold/reset configurations to compare
#    - nloop / nloop_train: total number of loops / last training loop
#    - cant_seed: number of independent network realizations per condition
# Target conditions
#    - Oscillations: choose frequency_values and amp0
#    - Sequences: choose sequence_amplitudes, sequence_sigmas, sequence_offsets, and number_sequences
# Parallelization
#    - parallelize_target_conditions, parallelize_pqif, parallelize_seed
#    - Normally only ONE of these should be > 1 at a time
# Data saving
#    - SAVE_ACTIVITY_LOOPS: loops for which full time-series data are saved
#    - SAVE_INPUTS: save total neuronal inputs
#    - SAVE_OUTPUTS: save neuronal outputs
#    - SAVE_NSPIKES: save spike trains
#    - SAVE_DW: track and save weight-change magnitude during learning

# Activity arrays are saved compressed to reduce disk usage.
# Continuous-valued arrays are stored as float32 and spikes as uint8.
# These conversions affect only saved data; simulation calculations retain
# their original numerical precision.

# Results are written to SAVE_DIR = SCRIPT_DIR / sim_name.
# An existing SAVE_DIR causes the script to abort to prevent overwriting data.

# Now also using jit for faster learning

# -------------------- Imports --------------------
from snn_core import dynamics, detect, evolution, initialize_neurons, initialize_connectivity_matrix, initialize_training, currents
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
# from scipy.stats import pearsonr
import pandas as pd
import csv
from joblib import Parallel, delayed
import time
import sys
import os
from pathlib import Path
from datetime import date
import json
from itertools import product
from numba import njit

# -------------------- Logging --------------------

# Included in order to both get print statements while the experiment runs, and a saved output file
class Tee:
    """ 
    Write output to multiple streams at once. Script prints to terminal and all prints are saved to an output file.
    """ 
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# -------------------- Initialize experiment --------------------

MASTER_SEED = int(os.environ.get("MASTER_SEED", "1102"))  # Master reproducibility seed
target_dynamics = "sequences"  # choose either oscillations or sequences
sigman_value = float(os.environ.get("SIM_SIGMAN", "0.1"))
current_amplitude_value = float(os.environ.get("SIM_CURRENT_AMPLITUDE", "8"))
current_low_frac = float(os.environ.get("SIM_CURRENT_LOW_FRAC", "0"))
current_high_frac = float(os.environ.get("SIM_CURRENT_HIGH_FRAC", "1"))
vreset_value = float(os.environ.get("SIM_VRESET", "-16"))
gsyn_value = float(os.environ.get("SIM_GSYN", "0.5"))
target_amplitude_value = float(os.environ.get("SIM_TARGET_AMPLITUDE", "8"))
cant_seed = int(os.environ.get("SIM_CANT_SEED", "10"))
parallelize_seed_value = int(os.environ.get("SIM_PARALLEL_SEED", str(cant_seed)))
if current_low_frac > current_high_frac:
    raise ValueError("SIM_CURRENT_LOW_FRAC must be <= SIM_CURRENT_HIGH_FRAC")
sim_name = os.environ.get(
    "SIM_NAME",
    f"{MASTER_SEED}_{target_dynamics}_noise_{sigman_value:g}_amp_{current_amplitude_value:g}"
    f"_current_{current_low_frac:g}_{current_high_frac:g}_vreset_{vreset_value:g}",
) # Name of this simulation, determines root folder and log file naming
SCRIPT_DIR = Path(__file__).resolve().parent  # Directory where this file is located, where also the generated files will be saved
SAVE_DIR = SCRIPT_DIR / sim_name  # This becomes the root directory for the simulations

if SAVE_DIR.exists():  # Do not overwrite if this master seed already exist
    raise FileExistsError(
        f"Simulation folder already exists: {SAVE_DIR}\n"
        "Aborting to prevent overwriting existing results."
    )
SAVE_DIR.mkdir(parents=True)



# -------------------- Set network and training parameters (shared for both target dynamics) --------------------
N = 200                 # Number of nodes (neurons)
N2 = int(N/2)           # Half
pqif_values = [0, 0.5, 1]  # fractions of QIF neurons in the network

# Synaptic connections
p = 0.3                 # Probability of connection (non-zero elements in the weight matrix)
# NOW: (CAMBIO) Make the initial synaptic strength configurable for parameter sweeps.
gsyn = gsyn_value       # Initial synaptic strength
alpha = 0.25            # Weight regularization parameter

# NOW: (CAMBIO) Keep one authoritative definition for each simulation parameter.
# Dynamics
dt = 0.1                # Time step (time scale 10 ms)
itmax = 1000            # Number of iterations, where 1000 --> 1 sec
sigman = sigman_value  # Noise standard deviation

# Stimulus
itstim = 200            # Stimulation time
current_amplitude = current_amplitude_value  # Stimulus intensity
# NOW: (CAMBIO) Configure the initial-current interval as amplitude * [low_frac, high_frac].
IEXT_LOW_FRAC = current_low_frac
IEXT_HIGH_FRAC = current_high_frac

# Training
nloop = 16              # Number of loops, 0: pre-training, last: post-training
nloop_train = 10        # Last training loop
cant_seed = cant_seed  # Number of independent replicates per experimental condition
ts = 5
b = 1 / ts              # Adaptation parameter for r
ftrain = 1              # Fraction of neurons to train

# Simulation configurations
configs = [
    # {'vt': 0, 'vreset': -30},  # Simulation 1
    # {'vt': 0, 'vreset': -25},  # Simulation 2
    # NOW: VRESET CHANGED FROM -21 TO -16.
    {'vt': 0, 'vreset': vreset_value},    # Simulation 3
    # {'vt': 0, 'vreset': -17},  # Simulation 4
]

# vrests = [-30, -25, -21, -17, -15]

# -------------------- Set parameters specifically for target dynamics --------------------

# OSCILLATIONS
amp0 = 4 # Target amplitude
frequency_values = [1, 3, 5] # Frequencies to combine

oscillation_parameter_combinations = [
    (omega1, omega2)
    for omega1 in frequency_values
    for omega2 in frequency_values
    if omega1 <= omega2 # 
]  

oscillation_conditions = [
    {
        "condition_id": condition_id,
        "r1": float(r1),
        "r2": float(r2),
    }
    for condition_id, (r1, r2)
    in enumerate(oscillation_parameter_combinations)
]

# SEQUENCES
# number_sequences = 2
# sequence_duration = dt * itmax / number_sequences
# sequence_amplitudes = np.array([0.6, 2.0, 6.0, 15.0])  # peak activity above baseline
# sequence_sigmas = np.array([1.0, 5.0, 15.0, 30.0])  # temporal SD of Gaussian activity peak, in target time units
# sequence_offsets = np.array([0.0, 0.25, 1.0, 2.0])  # baseline activity

number_sequences = 1
sequence_duration = dt * itmax / number_sequences


#with new vreset
# sequence_amplitudes = np.array([4.0, 8.0, 12.0, 16.0])
# sequence_sigmas = np.array([12.5, 15, 17.5, 20.0])
# sequence_offsets = np.array([0.0, 4.0, 5.0, 8.0])

# NOW: SINGLE TARGET CONDITION INSTEAD OF THE 4 x 4 x 3 GRID: AMPLITUDE 8, SIGMA 15, OFFSET 0.
# NOW: (CAMBIO) Make the sequence target amplitude configurable for parameter sweeps.
sequence_amplitudes = np.array([target_amplitude_value])
sequence_sigmas = np.array([15.0])
sequence_offsets = np.array([0.0])


sequence_parameter_combinations = list(product(sequence_amplitudes, sequence_sigmas, sequence_offsets,))  # Cartesian product of the sequence parameters

sequence_conditions = [
    # To simplify retrieval, each combination is associated with an integer id and the mapping is saved to file
    {
        "condition_id": condition_id,
        "amplitude": float(amplitude),
        "sigma": float(sigma),
        "offset": float(offset),
    }
    for condition_id, (
        amplitude,
        sigma,
        offset,
    ) in enumerate(sequence_parameter_combinations)
]



# -------------------- Parallelization --------------------
# Choose parallelization strategy. Only one level should normally be parallelized at a time. If set to 1, it becomes serial

# NOW: PARALLELIZATION MOVED TO THE SEED LEVEL (ONE JOB PER SEED). PQIF AND TARGET CONDITIONS RUN SERIALLY, SINCE ONLY ONE LEVEL SHOULD BE PARALLELIZED AT A TIME.
# NOW: (CAMBIO) Allow sweeps to disable seed-level parallel writes when needed.
parallelize_seed = parallelize_seed_value
parallelize_pqif = 1
parallelize_target_conditions = 1 # len(sequence_parameter_combinations)  # shared variable for both

n_parallel_levels = sum([
    parallelize_target_conditions > 1,
    parallelize_pqif > 1,
    parallelize_seed > 1,
])

if n_parallel_levels > 1:
    # Avoid parallelizing over multiple parameters
    raise ValueError(
        "Choose only one parallelization level: "
        "sequence conditions, pqif, or seeds."
    )


# -------------------- Choose saving --------------------
SAVE_ACTIVITY_LOOPS = [0, nloop_train + 1, nloop - 1]
SAVE_INPUTS = True
SAVE_OUTPUTS = True
SAVE_NSPIKES = False
SAVE_DW = False 

# -------------------- Save experiment config --------------------

def make_target_metadata():
    """
    Save target information. Is included in the metadata file generated in save_experiment_config()
    """

    target = {
        "type": target_dynamics,
    }

    if target_dynamics == "oscillations":
        target.update({
            "amp0": amp0,
            "frequency_values": frequency_values,
            "n_parameter_combinations": len(
                oscillation_parameter_combinations
            ),
        })

    elif target_dynamics == "sequences":
        target.update({
            "number_sequences": number_sequences,
            "sequence_duration": sequence_duration,
            "amplitudes": sequence_amplitudes.tolist(),
            "sigmas": sequence_sigmas.tolist(),
            "offsets": sequence_offsets.tolist(),
            "n_parameter_combinations": len(
                sequence_parameter_combinations
            ),
        })

    else:
        raise ValueError(
            f"Unknown target_dynamics: {target_dynamics}"
        )

    return target

def save_experiment_config():

    """
    Save all metadata associated with the experiment
    """

    metadata = {
        "name": sim_name,

        "random_seeds": {
            "target_seed": MASTER_SEED,
            "external_current_seed": MASTER_SEED + 100_000,
            "network_seed_rule": "MASTER_SEED + seed + 1",
        },

        "network": {
            "N": N,
            "p": p,
            "gsyn": gsyn,
            "alpha": alpha,
        },

        "dynamics": {
            "dt": dt,
            "itmax": itmax,
            "sigman": sigman,
            "ts": ts,
            "b": b,
        },

        "stimulus": {
            "itstim": itstim,
            "current_amplitude": current_amplitude,
            "current_low_frac": IEXT_LOW_FRAC,
            "current_high_frac": IEXT_HIGH_FRAC,
        },

        "training": {
            "nloop": nloop,
            "nloop_train": nloop_train,
            "ftrain": ftrain,
        },

        "target": make_target_metadata(),

        "conditions": {
            "pqif_values": pqif_values,
            "n_seeds": cant_seed,
        },

        "simulations": [
            {
                "sim": num_simulation,
                "vt": config["vt"],
                "vreset": config["vreset"],
            }
            for num_simulation, config
            in enumerate(configs, start=1)
        ],
    }

    config_path = SAVE_DIR / "experiment.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return config_path

# -------------------- File organization functions --------------------

def create_directory(
    num_simulation,
    base_dir,
):
    """
    Create one simulation directory and its output subdirectories.
    """

    folder_name = (
        base_dir
        / f"simulation_{num_simulation}"
    )

    folder_name.mkdir(
        parents=True,
        exist_ok=True,
    )

    subfolders = {
        "connectivity": (
            folder_name / "connectivity_matrix"
        ),
        "currents": (
            folder_name / "currents"
        ),
        "inputs": (
            folder_name / "inputs"
        ),
        "outputs": (
            folder_name / "outputs"
        ),
        "nspikes": (
            folder_name / "nspikes"
        ),
    }

    for path in subfolders.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    return folder_name, subfolders


# -------------------- Generate target patterns and save targets --------------------

def generate_oscillations_target(romega1, romega2, amp0, rng):
    '''
    Generates oscillatory target with frequencies romega1 and romega2
    With itmax 1000, the target makes one complete cycle over a loop when romega=1, and
    five complete cycles over a loop when romega=5
    '''
    target = np.zeros((N, itmax))

    # Using reproducible seed
    amp = rng.uniform(size=N) * amp0  # Individual amplitude for each neuron
    phase = rng.uniform(0, 2*np.pi, size=N)  # Phase
    indices = rng.permutation(N)  # Random ordering of neuron indices


    indices_romega1 = indices[:N2]
    indices_romega2 = indices[N2:]

    romega_vec = np.zeros(N)

    romega_vec[indices_romega1] = romega1
    romega_vec[indices_romega2] = romega2

    omega = romega_vec * 2 * np.pi / itmax

    for it in range(itmax):
        target[:, it] = amp * np.cos(it * omega + phase)

    return (
        target,
        amp,
        phase,
        omega,
        romega_vec
    )

def save_oscillation_target(target,phase,omega,romega_vec,amp,amp0,r1,r2,target_dir):
    """Save oscillatory target parameters"""
    target_parameters = pd.DataFrame({
        "neuron_index": np.arange(N),
        "frequency_pair": [f"({r1},{r2})"] * N,
        "romega_assigned": romega_vec,
        "omega_per_timestep": omega,
        "phase": phase,
        "amplitude": amp,
        "amp0": np.full(N, amp0),
    })

    target_parameters.to_csv(target_dir / "target_parameters.csv",index=False,)
    np.save(target_dir / "target_values.npy",target.astype(np.float32),)



def save_target(
    target,
    phase,
    omega,
    romega_vec,
    amp,
    amp0,
    r1,
    r2
):
    """
    Save target values and parameters.
    """

    target_parameters = pd.DataFrame({
        "neuron_index": np.arange(N),
        "frequency_pair": [f"({r1},{r2})"] * N,
        "romega_assigned": romega_vec,
        "omega_per_timestep": omega,
        "phase": phase,
        "amplitude": amp,
        "amp0": np.full(N, amp0),
    })

    target_parameters.to_csv(
        SAVE_DIR / "target_parameters.csv",
        index=False,
    )

    np.save(SAVE_DIR / "target_values.npy", target.astype(np.float32))



def generate_sequences_target(amplitude, sigma, offset, sequence_duration, rng):
    """
    Generate a periodic sequential target.

    Parameters
    ----------
    amplitude : float
        Difference between each neuron's peak firing rate and baseline firing rate.

    sigma : float
        Temporal standard deviation of each neuron's activity peak.

    offset : float
        Baseline firing rate.

    sequence_duration : float
        Duration of one complete sequence.

    rng : np.random.Generator
        Random-number generator used to assign sequence positions to neuron identities.
    """

    t_target = np.arange(itmax) * dt

    peak_times_by_position = np.arange(N) * sequence_duration / N

    neuron_permutation = rng.permutation(N)
    peak_times_by_neuron = peak_times_by_position[neuron_permutation]

    time_distance = (
        (t_target[None, :] - peak_times_by_neuron[:, None] + sequence_duration / 2)
        % sequence_duration
        - sequence_duration / 2
    )

    target = offset + amplitude * np.exp(-(time_distance ** 2) / (2 * sigma ** 2))

    return target, neuron_permutation, peak_times_by_position, peak_times_by_neuron, t_target

def save_sequence_target(
    target,
    neuron_permutation,
    peak_times_by_position,
    peak_times_by_neuron,
    t_target,
    amplitude,
    sigma,
    offset,
    sequence_duration,
    target_dir,
):

    target_parameters = pd.DataFrame({
        "neuron_index": np.arange(N),

        # position in the temporal sequence assigned to this neuron
        "sequence_position": neuron_permutation,

        # ordered peak corresponding to that position
        "peak_time": peak_times_by_neuron,

        "amplitude": np.full(N, amplitude),
        "sigma": np.full(N, sigma),
        "offset": np.full(N, offset),
        "sequence_duration": np.full(
            N,
            sequence_duration,
        ),
    })

    target_parameters.to_csv(
        target_dir / "target_parameters.csv",
        index=False,
    )

    np.save(target_dir / "target_values.npy", target.astype(np.float32))

    np.save(target_dir / "target_time.npy", t_target.astype(np.float32))

def save_neuron_assignment(
    pqif,
    folder_name
):
    nqif = int(N * pqif)

    neuron_type = np.full(N, "LIF", dtype=object)
    neuron_type[:nqif] = "QIF"

    assignment_df = pd.DataFrame({
        "neuron_index": np.arange(N),
        "neuron_model": neuron_type
    })

    assignment_df.to_csv(
        Path(folder_name)
        / f"neuron_assignment_pqif_{pqif}.csv",
        index=False,
    )




# def learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer, N, itmax, itstim, track_dw=SAVE_DW):
#     """
#     Recursive Least Squares (RLS) learning update.
#     """
#     error = target[:, it:it + 1] - w @ r.reshape(N, 1)

#     if track_dw:
#         w1 = np.zeros(w.shape)

#     for i in range(N):
#         ri = r[idx[i]].reshape(len(idx[i]), 1)
#         k1 = P[i] @ ri
#         k2 = ri.T @ P[i]
#         den = 1 + ri.T @ k1
#         P[i] -= (k1 @ k2) / den
#         dw = error[i, 0] * P[i] @ r[idx[i]]
#         w[i, idx[i]] += dw

#         if track_dw:
#             w1[i, idx[i]] += dw

#     if track_dw and dw_buffer is not None:
#         if it % 10 == 0:
#             modt_value = (it - itstim) + (iloop - 1) * (itmax - itstim)
#             modw_value = np.log(np.linalg.norm(w1) / norm_w0)
#             dw_buffer.append((modt_value, modw_value))

#     return w, P

@njit(cache=True)
def rls_update_numba(P_i, ri, error_i):
    """JIT-compiled RLS update for one neuron."""
    n = ri.shape[0]
    k = np.empty(n, dtype=np.float64)
    for j in range(n):
        value = 0.0
        for m in range(n):
            value += P_i[j, m] * ri[m]
        k[j] = value
    den = 1.0
    for j in range(n):
        den += ri[j] * k[j]
    inv_den = 1.0 / den
    for j in range(n):
        for m in range(n):
            P_i[j, m] -= k[j] * k[m] * inv_den
    dw = np.empty(n, dtype=np.float64)
    scale = error_i * inv_den
    for j in range(n):
        dw[j] = scale * k[j]
    return dw

# The new implementation uses the fact that P is symmetric, so r^TP=(Pr)^T, meaning we only need to calculate k=Pr
def learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer, N, itmax, itstim, track_dw=SAVE_DW):
    """Recursive Least Squares learning update with JIT-compiled inner RLS calculation."""
    prediction = np.asarray(w @ r.reshape(N, 1)).ravel()
    error = target[:, it] - prediction
    if track_dw:
        w1 = np.zeros(w.shape)
    for i in range(N):
        indices = idx[i]
        ri = r[indices]
        dw = rls_update_numba(P[i], ri, error[i])
        w[i, indices] += dw
        if track_dw:
            w1[i, indices] = dw
    if track_dw and dw_buffer is not None and it % 10 == 0:
        modt_value = (it - itstim) + (iloop - 1) * (itmax - itstim)
        modw_value = np.log(np.linalg.norm(w1) / norm_w0)
        dw_buffer.append((modt_value, modw_value))
    return w, P

# -------------------- Motifs and dimensionality calculations --------------------
            
def motifs(w,gsyn,N):
    w=w-np.mean(w)
    
    ww=np.matmul(w,w)
    wtw=np.matmul(w.T,w)
    wwt=np.matmul(w,w.T)
    
    sigma2=np.trace(wwt)/N
    
    tau_rec=np.trace(ww)
    tau_rec/=sigma2*N
    
    tau_div=np.sum(wwt)-np.trace(wwt)
    tau_div/=sigma2*N*(N-1)
    
    tau_con=np.sum(wtw)-np.trace(wtw)
    tau_con/=sigma2*N*(N-1)
    
    tau_chn=2*(np.sum(ww)-np.trace(ww))
    tau_chn/=sigma2*N*(N-1)
    
    return sigma2,tau_rec,tau_div,tau_con,tau_chn


# -------------------- Running simulations (seeds and pqif) --------------------

def run_single_seed(seed, pqif, num_simulation, vt, vreset, target, 
                    N, N2, p, gsyn, nloop, nloop_train,
                    dt, itmax, itstim, current_amplitude, alpha, sigman,
                    b, iout, folder_name, folders, Iext):
    """
    Run complete simulation for a single seed
    """
    rng = np.random.default_rng(MASTER_SEED + seed + 1)

    # Calculate nqif based on proportion of QIF neurons
    nqif = int(N * pqif)
    
    
    # dw saving
    dw_buffer = [] if SAVE_DW else None
    
    # Initialize network
    x, r, nspike = initialize_neurons(N, nqif, rng, vreset, vt)

    
    # Initialize connectivity
    w = initialize_connectivity_matrix(N, p, gsyn, rng)
    norm_w0 = np.linalg.norm(w)
    P, idx = initialize_training(N, w, alpha)
    
    # # Prepare file for weight evolution tracking
    # filename_dw = folder_name / f'simulation_{num_simulation}_dw_pqif_{pqif}_seed_{seed}.csv'
    
    # Storage for results across all loops
    seed_results = []
    currents_buffer = []
    
    # with open(filename_dw, mode='w', newline='') as file_dw:
    #     csv_writer_dw = csv.writer(file_dw)
    #     csv_writer_dw.writerow(['modt', 'modw'])
        
    # Main training loop
    for iloop in range(nloop):
        
        save_activity = iloop in SAVE_ACTIVITY_LOOPS

        outputs_loop = [] if (save_activity and SAVE_OUTPUTS) else None
        inputs_loop = [] if (save_activity and SAVE_INPUTS) else None
        nspikes_loop = [] if (save_activity and SAVE_NSPIKES) else None
        
        # Define output paths
        path_inputs = folders["inputs"] / f'simulation_{num_simulation}_inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        path_nspikes = folders["nspikes"] / f'simulation_{num_simulation}_nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        path_outputs = folders["outputs"] / f'simulation_{num_simulation}_outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        
        # Time evolution for this loop
        for it in range(itmax):
            nspike = np.zeros(N)
            
            x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt=vt, vreset=vreset, rng=rng, N=N, sigman=sigman)
            
            entrada = II + v
            
            # Accumulate data in memory (more efficient than writing each iteration)
            # Only retain data that will actually be written to disk
            if save_activity:

                if SAVE_OUTPUTS:
                    outputs_loop.append(rout)

                if SAVE_INPUTS:
                    inputs_loop.append(entrada)

                if SAVE_NSPIKES:
                    nspikes_loop.append(nspike)
                        
            # Record currents at specific time points in specific loops
            if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                currents_buffer.append([pqif, seed, iloop, it, 
                                        II[0], v[0], II[1], v[1], 
                                        II[N2+1], v[N2+1], II[N2+2], v[N2+2]])
            
            # Apply learning rule during training period
            if iloop > 0 and iloop <= nloop_train and int(it >= itstim):  # might be a solution
                w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer, N, itmax, itstim)
        
        # # Save all data for this loop (single write per loop)
        # if save_activity:
        #     # np.save(path_inputs, np.asarray(inputs_loop, dtype=np.float32))
        #     # np.save(path_nspikes, np.asarray(nspikes_loop, dtype=np.uint8))
        #     # np.save(path_outputs, np.asarray(outputs_loop, dtype=np.float32))
        #     # Compressed versions to save space, load like this: inputs = np.load(filename)["inputs"]
        #     np.savez_compressed(
        #         path_inputs.with_suffix(".npz"),
        #         inputs=np.asarray(inputs_loop, dtype=np.float32),
        #     )
        #     np.savez_compressed(
        #         path_nspikes.with_suffix(".npz"),
        #         nspikes=np.asarray(nspikes_loop, dtype=np.uint8),
        #     )
        #     np.savez_compressed(
        #         path_outputs.with_suffix(".npz"),
        #         outputs=np.asarray(outputs_loop, dtype=np.float32),
        #     )

        # -------------------- Save time series --------------------

        if save_activity:

            if SAVE_INPUTS:
                np.savez_compressed(
                    path_inputs.with_suffix(".npz"),
                    inputs=np.asarray(inputs_loop, dtype=np.float32),
                )

            if SAVE_NSPIKES:
                np.savez_compressed(
                    path_nspikes.with_suffix(".npz"),
                    nspikes=np.asarray(nspikes_loop, dtype=np.uint8),
                )

            if SAVE_OUTPUTS:
                np.savez_compressed(
                    path_outputs.with_suffix(".npz"),
                    outputs=np.asarray(outputs_loop, dtype=np.float32),
                )
    
        # # Calculate network motifs
        # sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)
        
        # Save weight matrix at specific loops
        if iloop == 0 or iloop == (nloop_train + 1):
            path_w_seed = folders["connectivity"] / f'simulation_{num_simulation}_connectivity_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'

            # NOW: (CAMBIO) Ensure the worker-local output directory exists before saving.
            path_w_seed.parent.mkdir(parents=True, exist_ok=True)
            np.save(path_w_seed, np.asarray(w, dtype=np.float32))

        
        # # Store results for this loop
        # seed_results.append([
        #     pqif,
        #     seed,
        #     iloop,
        #     sigma2,
        #     tau_rec,
        #     tau_div,
        #     tau_con,
        #     tau_chn,
        # ])

    filename_dw = folder_name / f"simulation_{num_simulation}_dw_pqif_{pqif}_seed_{seed}.npz"

    if SAVE_DW and dw_buffer:
        dw_array = np.asarray(dw_buffer)
        np.savez_compressed(filename_dw, modt=dw_array[:, 0].astype(np.int32), modw=dw_array[:, 1].astype(np.float32))
        
    return seed_results, currents_buffer


def run_pqif_simulation(pqif, num_simulation, vt, vreset, target, N, N2, p, gsyn, nloop, nloop_train, cant_seed, dt, itmax, itstim, current_amplitude, alpha, sigman, b, iout, Iext, folder_name, folders):
    """
    Run simulation for specific pqif value
    """

    print(f"Simulation {num_simulation} - Processing pqif = {pqif}")
    print(f"vt={vt}, vreset={vreset}")
    
    # Prepare currents file header
    path_currents_seed = folders["currents"] / (f"simulation_{num_simulation}_currents_pqif_{pqif}.csv")

    with open(path_currents_seed, mode='w', newline='') as file_:
        writer_ = csv.writer(file_)
        writer_.writerow(['pqif', 'seed', 'iloop', 'it', 'II_0', 'v_0', 
                        'II_1', 'v_1', 'II_N2+1', 'v_N2+1', 'II_N2+2', 'v_N2+2'])
    

    # -------------------- Seeds --------------------

    all_seed_results = []
    all_currents = []

    parallel_seed_results = Parallel(
        n_jobs=parallelize_seed,
        verbose=1
    )(
        delayed(run_single_seed)(
            seed, pqif, num_simulation, vt, vreset, target,
            N, N2, p, gsyn, nloop, nloop_train,
            dt, itmax, itstim, current_amplitude, alpha, sigman,
            b, iout, folder_name, folders, Iext
        )
        for seed in range(cant_seed)
    )

    for seed_results, currents_buffer in parallel_seed_results:
        all_seed_results.extend(seed_results)
        all_currents.extend(currents_buffer)

    # -------------------- Write results --------------------
    
    
    # Write all currents to file
    if all_currents:
        with open(path_currents_seed, 'a', newline='') as f_corr:
            writer_corr = csv.writer(f_corr)
            writer_corr.writerows(all_currents)
    
    print(f"pqif={pqif} completed for simulation {num_simulation}\n")
    
    return all_seed_results, all_currents


# -------------------- Run all network conditions for one target --------------------

def run_all_network_conditions(target, base_dir, Iext, iout):
    """
    Run all network configurations, pqif values, and seeds for one fixed target.

    Parameters
    ----------
    target : ndarray, shape (N, itmax)
        Target activity used for training.
    base_dir : Path
        Directory in which simulation_1, simulation_2, ... will be created.
        For oscillations: base_dir = SAVE_DIR
        For sequences: base_dir = SAVE_DIR / condition_XXX
    Iext : ndarray, shape (N, itmax)
        External current shared across experimental conditions.
    iout : ndarray
        Indices of neurons whose activity is recorded.
    """

    for num_simulation, config in enumerate(configs, start=1):
        vt = config["vt"]
        vreset = config["vreset"]

        print(f"\n{'-' * 70}")
        print(f"STARTING SIMULATION {num_simulation}: vt={vt}, vreset={vreset}")
        print(f"{'-' * 70}\n")

        folder_name, folders = create_directory(num_simulation=num_simulation, base_dir=base_dir)

        # -------------------- Motif results file --------------------
        # Uncomment if motifs are calculated during simulation
        #
        # csv_file_path = folder_name / f"simulation_{num_simulation}_results.csv"
        # with open(csv_file_path, "w", newline="") as f:
        #     csv.writer(f).writerow([
        #         "pqif", "seed", "iloop",
        #         "sigma2", "tau_rec", "tau_div", "tau_con", "tau_chn",
        #     ])

        # Save QIF/LIF neuron assignment for each pqif condition
        for pqif in pqif_values:
            save_neuron_assignment(pqif=pqif, folder_name=folder_name)

        # Run all pqif conditions
        parallel_results = Parallel(n_jobs=parallelize_pqif, verbose=1)(
            delayed(run_pqif_simulation)(
                pqif, num_simulation, vt, vreset, target,
                N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                dt, itmax, itstim, current_amplitude, alpha, sigman,
                b, iout, Iext, folder_name, folders,
            )
            for pqif in pqif_values
        )

        # -------------------- Collect and save motif results --------------------
        # Uncomment if motifs are calculated during simulation
        #
        # all_simulation_results = []
        # for pqif_seed_results, _ in parallel_results:
        #     all_simulation_results.extend(pqif_seed_results)
        #
        # with open(csv_file_path, "a", newline="") as file_res:
        #     csv.writer(file_res).writerows(all_simulation_results)

        print(f"SIMULATION {num_simulation} COMPLETED")


def run_oscillatory_condition(condition_id, r1, r2, Iext, iout):
    """Generate one oscillatory target and run all network conditions for that target"""

    n_conditions = len(oscillation_parameter_combinations)

    print(f"OSCILLATION CONDITION: {condition_id + 1}/{n_conditions}")
    print(f"condition_id: {condition_id}")
    print(f"r1: {r1}\nr2: {r2}")

    condition_dir = (SAVE_DIR / f"condition_{condition_id:03d}")
    condition_dir.mkdir(parents=True,exist_ok=True,)

    # Generate and save target
    target_rng = np.random.default_rng(MASTER_SEED)
    (target,amp,phase,omega,romega_vec,) = generate_oscillations_target(romega1=r1, romega2=r2, amp0=amp0, rng=target_rng,)

    save_oscillation_target(
        target=target,
        phase=phase,
        omega=omega,
        romega_vec=romega_vec,
        amp=amp,
        amp0=amp0,
        r1=r1,
        r2=r2,
        target_dir=condition_dir,
    )

    print("Oscillation target generated and saved.")

    # Run simulation
    run_all_network_conditions(target=target,base_dir=condition_dir,Iext=Iext,iout=iout,)
    print(f"OSCILLATION CONDITION {condition_id} COMPLETED")


def run_sequence_condition(
    condition_id,
    target_amplitude,
    target_sigma,
    target_offset,
    Iext,
    iout,
):
    """
    Generate one sequence target and run all network conditions
    for that target.
    """

    n_conditions = len(
        sequence_parameter_combinations
    )

    print("\n" + "=" * 70)
    print(
        f"SEQUENCE CONDITION "
        f"{condition_id + 1}/{n_conditions}"
    )
    print(f"condition_id : {condition_id}")
    print(f"amplitude    : {target_amplitude}")
    print(f"sigma        : {target_sigma}")
    print(f"offset       : {target_offset}")
    print("-" * 70)

    # Directory for this sequence condition
    condition_dir = (
        SAVE_DIR
        / f"condition_{condition_id:03d}"
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same sequence permutation for every condition
    target_rng = np.random.default_rng(
        MASTER_SEED
    )

    (
        target,
        neuron_permutation,
        peak_times_by_position,
        peak_times_by_neuron,
        t_target,
    ) = generate_sequences_target(
        amplitude=target_amplitude,
        sigma=target_sigma,
        offset=target_offset,
        sequence_duration=sequence_duration,
        rng=target_rng,
    )

    save_sequence_target(
        target=target,
        neuron_permutation=neuron_permutation,
        peak_times_by_position=peak_times_by_position,
        peak_times_by_neuron=peak_times_by_neuron,
        t_target=t_target,
        amplitude=target_amplitude,
        sigma=target_sigma,
        offset=target_offset,
        sequence_duration=sequence_duration,
        target_dir=condition_dir,
    )

    run_all_network_conditions(
        target=target,
        base_dir=condition_dir,
        Iext=Iext,
        iout=iout,
    )

    print(
        f"SEQUENCE CONDITION "
        f"{condition_id} COMPLETED"
    )

# -------------------- MAIN EXECUTION --------------------

if __name__ == "__main__":

    target_seed = MASTER_SEED  # every condition receives exactly the same neuron-to-sequence-position permutation
    current_seed = MASTER_SEED + 100_000  # Separate seed for external stimulus current


    # Files / logging
    log_path = (SAVE_DIR / f"{sim_name}.out")
    config_path = save_experiment_config()
    print(f"Saved experiment configuration to {config_path}")


    # Track runtime
    start_time = time.perf_counter()
    today = date.today()

    # Redirect stdout/stderr to terminal + log file
    log_fh = open(
        log_path,
        mode="w",
        encoding="utf-8",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(
        original_stdout,
        log_fh,
    )

    sys.stderr = Tee(
        original_stderr,
        log_fh,
    )

    # Record all neurons
    iout = np.linspace(
        0,
        N,
        num=N,
        endpoint=False,
    ).astype(int)

    try:

        # Print experiment information while running the script

        print("-" * 60)
        print(f"EXPERIMENT: {sim_name}")
        print("-" * 60)

        print(f"   date: {today}")
        print(
            f"   target dynamics: "
            f"{target_dynamics}"
        )
        print(
            f"   simulations: "
            f"{len(configs)}"
        )
        print(
            f"   pqif values: "
            f"{pqif_values}"
        )
        print(
            f"   seeds per condition: "
            f"{cant_seed}"
        )
        print(
            f"   parallel PQIF jobs: "
            f"{parallelize_pqif}"
        )
        print(
            f"   parallel seed jobs: "
            f"{parallelize_seed}"
        )

        if target_dynamics == "sequences":

            print(
                f"   sequence parameter conditions: "
                f"{len(sequence_parameter_combinations)}"
            )

            print(
                f"   sequences per target: "
                f"{number_sequences}"
            )

            print(
                f"   sequence duration: "
                f"{sequence_duration}"
            )
        elif target_dynamics == "oscillations":

            print(
                f"   oscillation frequency values: "
                f"{frequency_values}"
            )

            print(
                f"   oscillation parameter conditions: "
                f"{len(oscillation_parameter_combinations)}"
            )

            print(
                f"   parallel oscillation jobs: "
                f"{parallelize_target_conditions}"
            )

        print("-" * 60)


        # Generate external current ONCE

        print("\nGenerating shared external current...")

        current_rng = np.random.default_rng(current_seed)

        Iext = currents(N,itmax,current_rng,current_amplitude, IEXT_LOW_FRAC, IEXT_HIGH_FRAC, itstim)

        np.save(SAVE_DIR / "external_current.npy", Iext.astype(np.float32))

        print("External current generated and saved.")


        # -------------------- TARGET --------------------

        if target_dynamics == "oscillations":

            print(
                "\nGenerating oscillation target..."
            )

            # target_rng = np.random.default_rng(
            #     target_seed
            # )

            # (target,amp,phase,omega,romega_vec,) = generate_oscillations_target(
            #     romega1=r1,
            #     romega2=r2,
            #     amp0=amp0,
            #     rng=target_rng,
            # )

            # save_target(
            #     target=target,
            #     phase=phase,
            #     omega=omega,
            #     romega_vec=romega_vec,
            #     amp=amp,
            #     amp0=amp0,
            #     r1=r1,
            #     r2=r2,
            # )

            # print("Oscillation target generated and saved.")

            # One oscillation target is shared across all configs / pqif values / seeds.
            # run_all_network_conditions(
            #     target=target,
            #     base_dir=SAVE_DIR,
            #     Iext=Iext,
            #     iout=iout,
            # )
            oscillation_conditions_df = pd.DataFrame(oscillation_conditions)
            oscillation_conditions_df.to_csv(SAVE_DIR / "oscillation_conditions.csv",index=False,)

            Parallel(
                n_jobs=parallelize_target_conditions,
                verbose=1,
            )(
                delayed(run_oscillatory_condition)(
                    condition_id,
                    r1,
                    r2,
                    Iext,
                    iout,
                )
                for condition_id, (r1, r2)
                in enumerate(oscillation_parameter_combinations)
            )



        elif target_dynamics == "sequences":

            sequence_conditions_df = pd.DataFrame(sequence_conditions)  # Save mapping of integer id to parameter configuration
            sequence_conditions_df.to_csv(SAVE_DIR / "sequence_conditions.csv",index=False,)

            print("\nSaved sequence condition table.")

            Parallel(
                n_jobs=parallelize_target_conditions,
                verbose=1,
            )(
                delayed(run_sequence_condition)(
                    condition_id,
                    target_amplitude,
                    target_sigma,
                    target_offset,
                    Iext,
                    iout,
                )
                for condition_id, (
                    target_amplitude,
                    target_sigma,
                    target_offset,
                ) in enumerate(
                    sequence_parameter_combinations
                )
            )

        else:
            # Invalid target type
            raise ValueError(
                "No valid target dynamics was provided. "
                "Valid dynamics are "
                "'oscillations' or 'sequences'."
            )


        # Experiment finished

        print("\n"+ "-" * 60)

        print("ALL SIMULATIONS COMPLETED SUCCESSFULLY")

        elapsed_time = (time.perf_counter() - start_time)

        hours, remainder = divmod(elapsed_time,3600,)

        minutes, seconds = divmod(remainder,60,)

        print(
            f"Total time taken: "
            f"{int(hours):02d}:"
            f"{int(minutes):02d}:"
            f"{seconds:05.2f} "
            f"(hh:mm:ss)"
        )

        print(
            f"Output log saved to: "
            f"{log_path}"
        )

        print("-" * 60)

    finally:

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        log_fh.close()