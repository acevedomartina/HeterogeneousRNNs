#  -------------------- Run simulations --------------------

# Choose whether the network is trained on oscillations or sequences at the top of the script.
# Choose parameters and parallelization strategy under "Global parameters and initialization"

# Target is shared across seeds.
# External stimulus current (the first itstim steps) is shared across seeds

# Everywhere sampling/randomness happens, it is routed through an explicit rng.
# Using the same master seed should reproduce the same random draws and simulation results.

# 32 bit storage some places (e.g. modt, modw), might change this later if we need 64-bit precision

# Vary parameters of sequences:
# The amplitude took values between 0.2 and 1, the width between 1 and 1.9, and obset between 0 and 0.45.

# For each sequence condition, it generates one target, saves it and call run_all_network_conditions(...)
# target_rng is created from the same seed for every sequence condition, so permutation is identical across all targets
# Iext is generated only once and reused everywhere

# Therefore, when comparing two sequence-condition folders, the intended experimental difference is the sequence parameters rather than a different sequence ordering, stimulus, or initial network realization.


# -------------------- Imports --------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
# from scipy.stats import pearsonr
import pandas as pd
import csv
from joblib import Parallel, delayed
import time
import sys
from pathlib import Path
from datetime import date
import json
from itertools import product

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

MASTER_SEED = 1008  # Master reproducibility seed
target_dynamics = "oscillations"  # choose either oscillations or sequences
sim_name = F"{MASTER_SEED}_{target_dynamics}" # Name of this simulation, determines root folder and log file naming
SCRIPT_DIR = Path(__file__).resolve().parent  # Directory where this file is located, where also the generated files will be saved
SAVE_DIR = SCRIPT_DIR / sim_name  # This becomes the root directory for the simulations
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# PARAMETERS:
# Neuron
N = 200                 # Number of nodes (neurons)
N2 = int(N/2)           # Half
pqif_values = [0, 0.5, 1]  # fractions of QIF neurons in the network

# Synaptic connections
p = 0.3                 # Probability of connection (non-zero elements in the weight matrix)
gsyn = 0.5              # Initial synaptic strength
alpha = 0.25            # Weight regularization parameter

# Dynamics
dt = 0.1                # Time step (time scale 10 ms)
itmax = 1000            # Number of iterations, where 1000 --> 1 sec
sigman = 1              # Noise standard deviation --> Noise in the dynamics

# Stimulus
itstim = 200            # Stimulation time
current_amplitude = 5   # Stimulus intensity

# Training
nloop = 16              # Number of loops, 0: pre-training, last: post-training
nloop_train = 10        # Last training loop
cant_seed = 2          # Number of independent replicates per experimental condition 
ts = 5                  # 
b = 1 / ts              # adaptation parameter for r - In evolution of r, dr/dt = -b * r, b is magnitude
ftrain = 1              # Fraction of neurons to train


# Dynamics
dt = 0.1
itmax = 1000
sigman = 1

# Stimulus
itstim = 200
current_amplitude = 5

# Training
nloop = 16
nloop_train = 10

# Oscillations parameters
amp0 = 4                # Target amplitude
# Frequencies
frequency_values = [1, 3, 5]

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

# Sequences parameters
number_sequences = 1
sequence_duration = dt * itmax / number_sequences

sequence_amplitudes = np.array([
    2.0,
    4.0,   # old amp0
    6.0,
])

sequence_width_fractions = np.array([
    0.10,
    0.15,  # old sg_index
    0.20,
])

sequence_sigmas = sequence_width_fractions * sequence_duration  # scales with duration

sequence_offsets = np.array([
    0.0,   # old baseline
    1.0,
    2.0,
])

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


# Simulation configurations
configs = [
    {'vt': 0, 'vreset': -8.5},  # Simulation 1
    {'vt': 0, 'vreset': -12.3},  # Simulation 2
    {'vt': 0, 'vreset': -17},    # Simulation 3
    {'vt': 0, 'vreset': -22},  # Simulation 4
]

# Choose parallelization strategy.
# Only one level should normally be parallelized at a time.
# If set to 1, it becomes serial

parallelize_seed = 1
parallelize_pqif = 1
# parallelize_sequence_conditions = 10
# parallelize_oscillation_conditions = 1

parallelize_target_conditions = 10

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
    np.save(target_dir / "target_values.npy",target,)



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

    np.save(
        SAVE_DIR / "target_values.npy",  # is saved as shape (N, itmax)
        target
    )



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

    np.save(
        target_dir / "target_values.npy",
        target,
    )

    np.save(
        target_dir / "target_time.npy",
        t_target,
    )

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


# -------------------- Dynamics and learning functions --------------------

def dynamics(x_var,r_var,I_var,nqif, b, rng):
    '''
    Dynamics of neurons.
    Compute derivatives of neuron state and adaptation variable.
    LIF neurons: neurons from index nqif onward
    QIF neurons: neurons from index 0 to nqif-1

    LIF dynamics: dx/dt = -x + I + noise
    QIF dynamics: dx/dt = 1 - cos(x) + I*(1 + cos(x)) + noise
    QIF is a phase-based neuron model, x represents phase in [0, 2π)

    Adaptation variable r decays exponentially: dr/dt = -b * r
    
    Inputs:
        x_var   : internal state of neurons
        r_var   : output firing rate or adaptation variable
        I_var   : total input to neurons (external + recurrent)
        nqif    : number of QIF neurons at the start of the array
        b       : adaptation parameter for r
        
    Outputs:
        dx      : derivative of neuron state
        dr      : derivative of adaptation/firing rate
    '''
    # Initialize dx (derivative of state) as zeros for all neurons
    dx=np.zeros(N)

    # Add stochastic noise to inputs (generates Gaussian noise for LIF neurons)
    # I_noise_lif = np.random.randn(N - nqif)*sigman 

    # Generate Gaussian noise for QIF neurons  
    # I_noise_qif = np.random.randn(nqif)*sigman

    I_noise_lif = rng.standard_normal(N - nqif) * sigman
    I_noise_qif = rng.standard_normal(nqif) * sigman

    # Compute derivative for LIF neurons
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif

    # Compute derivative for QIF neurons
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif])) + I_noise_qif
    
    # Compute derivative for adaptation variable r
    dr = -b*r_var 

    return dx,dr

def detect(x, xnew, rnew, nspike, nqif, b, vt, vreset):

    # LIF neurons
    ispike_lif = np.where(
        (x[nqif:] < vt)
        & (xnew[nqif:] > vt)
    )[0]

    ispike_lif = ispike_lif + nqif  # shift index

    if len(ispike_lif) > 0:
        rnew[ispike_lif] += b
        xnew[ispike_lif] = vreset
        nspike[ispike_lif] += 1

    # QIF neurons
    dpi = np.mod(
        np.pi - np.mod(x[:nqif], 2 * np.pi),
        2 * np.pi,
    )

    delta_x = xnew[:nqif] - x[:nqif]

    ispike_qif = np.where(
        (delta_x > 0)
        & ((delta_x - dpi) > 0)
    )[0]

    if len(ispike_qif) > 0:
        rnew[ispike_qif] += b
        nspike[ispike_qif] += 1

    return xnew, rnew, nspike

def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vreset, rng):
    II = np.squeeze(np.asarray(Iext[:, it]))
    v = w.dot(r.T).A1
    dx, dr = dynamics(x, r, II + v, nqif, b, rng)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr = dynamics(xnew, rnew, II + v, nqif, b, rng)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vreset)
    x, r = np.copy(xnew), np.copy(rnew)

    return x, r, nspike, r[iout], II, v


def initialize_connectivity_matrix(N, p, gsyn, rng):
    # w = sparse.random(N, N, p, data_rvs=np.random.randn).todense()
    w = sparse.random(
        N,
        N,
        density=p,
        random_state=rng,
        data_rvs=lambda n: rng.standard_normal(n),
    ).todense()

    np.fill_diagonal(w, 0)  # No autapses
    w *= gsyn / np.sqrt(p * N)  # rescale each weight
    
    for i in range(N):
        i0 = np.where(w[i, :])[1]  # find j neurons that project to i
        if len(i0) > 0:
            av0 = np.sum(w[i, i0]) / len(i0)
            w[i, i0] -= av0  # Subtract mean so each row has zero mean
    
    return w

def initialize_neurons(N, rng):
    '''
    Initializes neurons

    Parameters
    --------------------
    N : Number of neurons

    Returns
    ----------
    x       : neuron internal state vector 
    r       : neuron output
    nspike  : initialized container for the spikes to N neurons

    '''
    x = rng.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    nspike = np.zeros(N)
    return x, r, nspike

def initialize_training(N, w):
    # Initialize correlation matrices for RLS learning
    nind=np.zeros(N).astype('int')
    idx=[]
    P=[]
    for i in range(N):
        ind=np.where(w[i,:])[1]
        nind[i]=len(ind)
        idx.append(ind)
        P.append(np.identity(nind[i])/alpha)   
    return P, idx

def currents(N, itmax, rng):
    Iext=np.zeros((N,itmax))
    # Ibac=current_amplitude*(2*np.random.uniform(size=N)-1)
    Ibac = current_amplitude * (2*rng.uniform(size=N)-1)
    Iext[:, :itstim] = Ibac[:, None]  # Vectorized assignment
    return Iext

def learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer, N, itmax, itstim, track_dw=False):
    """
    Recursive Least Squares (RLS) learning update.
    """
    error = target[:, it:it + 1] - w @ r.reshape(N, 1)

    if track_dw:
        w1 = np.zeros(w.shape)

    for i in range(N):
        ri = r[idx[i]].reshape(len(idx[i]), 1)
        k1 = P[i] @ ri
        k2 = ri.T @ P[i]
        den = 1 + ri.T @ k1
        P[i] -= (k1 @ k2) / den
        dw = error[i, 0] * P[i] @ r[idx[i]]
        w[i, idx[i]] += dw

        if track_dw:
            w1[i, idx[i]] += dw

    if track_dw and dw_buffer is not None:
        if it % 10 == 0:
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
    dw_buffer = []
    
    # Initialize network
    x, r, nspike = initialize_neurons(N, rng)

    
    # Initialize connectivity
    w = initialize_connectivity_matrix(N, p, gsyn, rng)
    norm_w0 = np.linalg.norm(w)
    P, idx = initialize_training(N, w)
    
    # Prepare file for weight evolution tracking
    filename_dw = folder_name / f'simulation_{num_simulation}_dw_pqif_{pqif}_seed_{seed}.csv'
    
    # Storage for results across all loops
    seed_results = []
    currents_buffer = []
    
    # with open(filename_dw, mode='w', newline='') as file_dw:
    #     csv_writer_dw = csv.writer(file_dw)
    #     csv_writer_dw.writerow(['modt', 'modw'])
        
    # Main training loop
    for iloop in range(nloop):
        
        # Pre-allocate arrays for this loop
        outputs_loop = []
        inputs_loop = []
        nspikes_loop = []
        
        # Define output paths
        path_inputs = folders["inputs"] / f'simulation_{num_simulation}_inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        path_nspikes = folders["nspikes"] / f'simulation_{num_simulation}_nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        path_outputs = folders["outputs"] / f'simulation_{num_simulation}_outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'
        
        # Time evolution for this loop
        for it in range(itmax):
            nspike = np.zeros(N)
            
            x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt=vt, vreset=vreset, rng=rng)
            
            entrada = II + v
            
            # Accumulate data in memory (more efficient than writing each iteration)
            outputs_loop.append(rout)
            inputs_loop.append(entrada)
            nspikes_loop.append(nspike)
            
            # Record currents at specific time points in specific loops
            if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                currents_buffer.append([pqif, seed, iloop, it, 
                                        II[0], v[0], II[1], v[1], 
                                        II[N2+1], v[N2+1], II[N2+2], v[N2+2]])
            
            # Apply learning rule during training period
            if iloop > 0 and iloop <= nloop_train and int(it >= itstim):  # might be a solution
                w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer, N, itmax, itstim, track_dw=False)
        
        # Save all data for this loop (single write per loop)
        np.save(path_inputs, np.asarray(inputs_loop))
        np.save(path_nspikes, np.asarray(nspikes_loop))
        np.save(path_outputs, np.asarray(outputs_loop))
        
        # Calculate network motifs
        sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)
        
        # Save weight matrix at specific loops
        if iloop == 0 or iloop == (nloop_train + 1):
            path_w_seed = folders["connectivity"] / f'simulation_{num_simulation}_connectivity_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy'

            np.save(path_w_seed, np.asarray(w))

        
        # Store results for this loop
        seed_results.append([
            pqif,
            seed,
            iloop,
            sigma2,
            tau_rec,
            tau_div,
            tau_con,
            tau_chn,
        ])

    filename_dw = folder_name / f"simulation_{num_simulation}_dw_pqif_{pqif}_seed_{seed}.npz"

    if dw_buffer:
        dw_array = np.asarray(dw_buffer)

        np.savez_compressed(
            filename_dw,
            modt=dw_array[:, 0].astype(np.int32),
            modw=dw_array[:, 1].astype(np.float32),
        )
    
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

def run_all_network_conditions(
    target,
    base_dir,
    Iext,
    iout,
):
    """
    Run all network configurations, pqif values, and seeds for one fixed target.

    Parameters
    ----------
    target : ndarray, shape (N, itmax)
        Target activity used for training.

    base_dir : Path
        Directory in which simulation_1, simulation_2, ...
        will be created.

        For oscillations:
            base_dir = SAVE_DIR

        For sequences:
            base_dir = SAVE_DIR / condition_XXX

    Iext : ndarray, shape (N, itmax)
        External current shared across experimental conditions.

    iout : ndarray
        Indices of neurons whose activity is recorded.
    """


    # Iterate over vt/vreset network configurations

    for num_simulation, config in enumerate(
        configs,
        start=1,
    ):

        vt = config["vt"]
        vreset = config["vreset"]

        print(f"\n{'-' * 70}")
        print(
            f"STARTING SIMULATION {num_simulation}: "
            f"vt={vt}, vreset={vreset}"
        )
        print(f"{'-' * 70}\n")


        # Create folders for this network configuration

        folder_name, folders = create_directory(
            num_simulation=num_simulation,
            base_dir=base_dir,
        )


        # Results file

        filename_results = (
            f"simulation_{num_simulation}_results.csv"
        )

        csv_file_path = (
            folder_name / filename_results
        )

        with open(
            csv_file_path,
            "w",
            newline="",
        ) as f:

            csv.writer(f).writerow([
                "pqif",
                "seed",
                "iloop",
                "sigma2",
                "tau_rec",
                "tau_div",
                "tau_con",
                "tau_chn",
            ])


        # Save which neurons are QIF/LIF for each pqif condition

        for pqif in pqif_values:

            save_neuron_assignment(
                pqif=pqif,
                folder_name=folder_name,
            )


        # Run all pqif conditions

        parallel_results = Parallel(
            n_jobs=parallelize_pqif,
            verbose=1,
        )(
            delayed(run_pqif_simulation)(
                pqif,
                num_simulation,
                vt,
                vreset,
                target,
                N,
                N2,
                p,
                gsyn,
                nloop,
                nloop_train,
                cant_seed,
                dt,
                itmax,
                itstim,
                current_amplitude,
                alpha,
                sigman,
                b,
                iout,
                Iext,
                folder_name,
                folders,
            )
            for pqif in pqif_values
        )

        # Collect pqif results

        all_simulation_results = []

        for (
            pqif_seed_results,
            pqif_currents,
        ) in parallel_results:

            all_simulation_results.extend(
                pqif_seed_results
            )


        # Save combined motif results

        with open(
            csv_file_path,
            "a",
            newline="",
        ) as file_res:

            writer_res = csv.writer(
                file_res
            )

            writer_res.writerows(
                all_simulation_results
            )

        print(
            f"SIMULATION {num_simulation} COMPLETED"
        )

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

        Iext = currents(N,itmax,current_rng,)

        np.save(SAVE_DIR / "external_current.npy",Iext,)

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