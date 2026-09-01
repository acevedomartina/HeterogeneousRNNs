#  -------------------- Run simulations --------------------

# Choose whether the network is trained on oscillations or sequences at the top of the script.
# Choose parameters and parallelization strategy under "Global parameters and initialization"

# Target is shared across seeds.
# External stimulus current (the first itstim steps) is shared across seeds

# Everywhere sampling/randomness happens, it is routed through an explicit rng.
# Using the same master seed should reproduce the same random draws and simulation results.

# 32 bit storage some places (e.g. modt, modw), might change this later if we need 64-bit precision

# ------------------------------------------------------------

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

# For logging (script prints to terminal and all prints are saved to an output file)
class Tee:
    """ 
    Write output to multiple streams at once.
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

MASTER_SEED = 1003  # For reproducibility

target_dynamics = "oscillations"  # choose either oscillations or sequences
sim_name = F"{MASTER_SEED}_{target_dynamics}" # Name of this simulation, determines root folder and log file naming
SCRIPT_DIR = Path(__file__).resolve().parent  # Directory where this file is located, where also the generated files will be saved
SAVE_DIR = SCRIPT_DIR / sim_name  # This becomes the root directory for the simulations
SAVE_DIR.mkdir(parents=True, exist_ok=True)



# -------------------- Global parameters and initialization --------------------
# Neurons
N = 200                 # Number of nodes (neurons)
N2 = int(N/2)           # Half
pqif_values = [0, 0.25, 0.5, 0.75, 1]  # fractions of QIF neurons in the network

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

# Target parameter shared by both target types
amp0 = 4                # Amplitude used in target

# Target parameters for oscillations
r1 = 5  # frequency 1
r2 = 5  # frequency 2

# Target parameters for sequences
sg_index=0.15
omegagauss=0.2

# Config
configs = [
    {'vt': 0, 'vreset': -8.5},  # Simulation 1
    {'vt': 0, 'vreset': -12.3},  # Simulation 2
    {'vt': 0, 'vreset': -17},    # Simulation 3
    {'vt': 0, 'vreset': -22},  # Simulation 4
]

# Choose parallelization strategy. If set to 1, it becomes serial
# Ideally keep one of them serial and the other parallelized
parallelize_seed = cant_seed
parallelize_pqif = 1  # len(pqif_values) if parallel

# -------------------- Save experiment config --------------------

def make_target_metadata():
    target = {
        "type": target_dynamics,
        "amp0": amp0,
    }

    if target_dynamics == "oscillations":
        target.update({
            "r1": r1,
            "r2": r2,
        })

    elif target_dynamics == "sequences":
        target.update({
            "sg_index": sg_index,
            "omegagauss": omegagauss,
        })

    else:
        raise ValueError(
            f"Unknown target_dynamics: {target_dynamics}"
        )

    return target

def save_experiment_config():

    metadata = {
        "name": sim_name,
        "master_seed": MASTER_SEED,
        "seed_rule": "MASTER_SEED + seed + 1",
        "shared_target_current_seed": MASTER_SEED,

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

def create_directory(num_simulation):
    """
    Create one simulation directory and its output subdirectories.
    """

    folder_name = SAVE_DIR / f"simulation_{num_simulation}"
    folder_name.mkdir(parents=True, exist_ok=True)

    # subfolder_activity = folder_name / "activity_examples"  # might be removed, as currently we do not use it
    subfolders = {
        "connectivity": folder_name / "connectivity_matrix",
        "currents": folder_name / "currents",
        "inputs": folder_name / "inputs",
        "outputs": folder_name / "outputs",
        "nspikes": folder_name / "nspikes",
    }

    for path in subfolders.values():
        path.mkdir(parents=True, exist_ok=True)

    return folder_name, subfolders


# Can remove this now because we keep everything in the .json

# def create_parameters_file(
#     filename_results,
#     num_simulation,
#     folder_name,
#     b,
#     vt,
#     vreset,
#     pqif_values,
#     target_dynamics,
#     r1=None,
#     r2=None,
#     sg_index=None,
#     omegagauss=None,
# ):
#     """
#     Save general simulation parameters and target-specific parameters.
#     """

#     data_parameters = {
#         "simulation_number": [num_simulation],
#         "target_dynamics": [target_dynamics],
#         "master_seed": [MASTER_SEED],

#         "N": [N],
#         "N2": [N2],
#         "p": [p],
#         "gsyn": [gsyn],
#         "alpha": [alpha],

#         "dt": [dt],
#         "itmax": [itmax],
#         "sigman": [sigman],

#         "itstim": [itstim],
#         "current_amplitude": [current_amplitude],

#         "nloop": [nloop],
#         "nloop_train": [nloop_train],
#         "cant_seed": [cant_seed],
#         "ts": [ts],
#         "b": [b],
#         "ftrain": [ftrain],

#         "vt": [vt],
#         "vreset": [vreset],

#         "pqif_values": [",".join(map(str, pqif_values))],

#         "results_file": [filename_results],
#         "target_parameters_file": ["../target_parameters.csv"],
#         "target_values_file": ["../target_values.npy"],
#         "external_current_file": ["../external_current.npy"],
#     }

#     # Target-specific parameters
#     if target_dynamics == "oscillations":
#         data_parameters.update({
#             "r1": [r1],
#             "r2": [r2],
#             "amp0": [amp0],
#         })

#     elif target_dynamics == "sequences":
#         data_parameters.update({
#             "sg_index": [sg_index],
#             "omegagauss": [omegagauss],
#             "amp0": [amp0],
#         })

#     else:
#         raise ValueError(
#             f"Unknown target_dynamics: {target_dynamics}"
#         )

#     df = pd.DataFrame(data_parameters)

#     filename_parametros = (
#         f"simulation_{num_simulation}_parameters.csv"
#     )

#     df.to_csv(
#         Path(folder_name) / filename_parametros,
#         index=False,
#     )

# -------------------- Generate target patterns and save target --------------------

def generate_oscillations_target(romega1, romega2, amp0, rng):
    '''
    Generates oscillatory target with frequencies romega1 and romega2
    With itmax 1000, the target makes one complete cycle over a loop when romega=1, and
    five complete cycles over a loop when romega=5
    ----------
    romega1:
    romega2: 
    amp0: 
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

    # target_df = pd.DataFrame(
    #     target.T,
    #     columns=[f"neuron_{i}" for i in range(N)],
    # )

    # target_df.to_csv(
    #     SAVE_DIR / "target_values.csv",
    #     index=False,
    # )

    np.save(
        SAVE_DIR / "target_values.npy",  # is saved as shape (N, itmax)
        target
    )

def generate_sequences_target(sg_index, omegagauss, amp0, rng):
    """
    Generate a sequential target by shifting a Gaussian activity profile
    across the network over time.

    Parameters
    ----------
    sg_index : float
        Gaussian width expressed as a fraction of network size.

    omegagauss : float
        Speed of the Gaussian shift in neuron-index units per timestep.

    amp0 : float
        Peak amplitude of the Gaussian profile.

    Returns
    -------
    target : ndarray, shape (N, itmax)
        Sequential target activity for all neurons and timesteps.

    neuron_permutation : ndarray, shape (N,)
        Random assignment between Gaussian sequence positions and neuron indices.

    gaussian_profile : ndarray, shape (N,)
        Original unshifted Gaussian activity profile.

    sg : float
        Gaussian width in neuron-index units.
    """

    target = np.zeros((N, itmax))

    # Gaussian width in neuron-index units
    sg = sg_index * N

    # Gaussian activity profile
    gaussian_profile = np.zeros(N)

    for i in range(N):
        gaussian_profile[i] = (amp0 * np.exp(-(i - N / 2) ** 2 / (2 * sg ** 2)))

    # Randomly assign sequence positions to neuron identities
    neuron_permutation = rng.permutation(N)

    # Shift the Gaussian profile over time
    for it in range(itmax):
        gaussian_shifted = np.roll(
            gaussian_profile,
            int(omegagauss * it)
        )

        target[:, it] = gaussian_shifted[neuron_permutation]

    return (
        target,
        neuron_permutation,
        gaussian_profile,
        sg,
    )

def save_sequence_target(
    target,
    neuron_permutation,
    gaussian_profile,
    sg,
    sg_index,
    omegagauss,
    amp0,
):
    """
    Save sequential target values and the parameters defining the sequence.
    Sequence position of neuron i = neuron_permutation[i]
    """

    target_parameters = pd.DataFrame({
        "neuron_index": np.arange(N),
        "sequence_position": neuron_permutation,
        "sg_index": np.full(N, sg_index),
        "sg": np.full(N, sg),
        "omegagauss": np.full(N, omegagauss),
        "amp0": np.full(N, amp0),
    })

    target_parameters.to_csv(
        SAVE_DIR / "target_parameters.csv",
        index=False,
    )

    # target_df = pd.DataFrame(
    #     target.T,
    #     columns=[f"neuron_{i}" for i in range(N)],
    # )

    # target_df.to_csv(
    #     SAVE_DIR / "target_values.csv",
    #     index=False,
    # )
    np.save(
        SAVE_DIR / "target_values.npy",
        target,
    )

    # gaussian_df = pd.DataFrame({
    #     "sequence_position": np.arange(N),
    #     "gaussian_amplitude": gaussian_profile,
    # })

    # gaussian_df.to_csv(
    #     # Original Gaussian amplitude at each sequence position
    #     SAVE_DIR / "sequence_profile.csv",
    #     index=False,
    # )

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

# -------------------- Helper --------------------


# def save_matrix_csv(matrix, file_name):
#     ''' 
#     Make matrix
#     '''
#     with open(file_name, 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         for row in matrix:
#             list_row = [str(element) for element in row.flat]  # rows
#             csv_writer.writerow(list_row)


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


# def detect(x,xnew,rnew,nspike,nqif, b, vt, vreset):

#     # LIF spike detection
#     ispike_lif=np.where(x[nqif:]<vt) and np.where(xnew[nqif:]>vt)
#     ispike_lif=ispike_lif[0]+nqif
#     if(len(ispike_lif)>0):
#         rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
#         xnew[ispike_lif[:]] = vreset
#         nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1

#     # QIF spike detection
#     dpi=np.mod(np.pi - np.mod(x,2*np.pi),2*np.pi)  # distance to pi
#     ispike_qif=np.where((xnew[:nqif]-x[:nqif])>0) and np.where((xnew[:nqif]-x[:nqif]-dpi[:nqif])>0)
#     if(len(ispike_qif)>0):
#         rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
#         nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
#     return xnew,rnew,nspike


# Possibly change to this detect function???
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
    # x = np.random.uniform(size=N) * 2 * np.pi
    # r = np.zeros(N)
    # nspike = np.zeros(N)
    # return x, r, nspike
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


def learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer):
    error = target[:, it:it + 1] - w @ r.reshape(N, 1)
    for i in range(N):
        ri = r[idx[i]].reshape(len(idx[i]), 1)  
        k1 = P[i] @ ri
        k2 = ri.T @ P[i]
        den = 1 + ri.T @ k1
        P[i] -= (k1 @ k2) / den
        dw = error[i, 0] * P[i] @ r[idx[i]]
        w[i, idx[i]] += dw

    if it % 10 == 0:
        modt_value = it + iloop * itmax
        modw_value = np.log(np.linalg.norm(w) / norm_w0)

        # file writing
        # csv_writer.writerow([modt_value, modw_value])
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
            # if iloop > 0 and iloop <= nloop_train and int(it > itstim):  # external stimulus ends at it=199, and itstim is 200. So this skips one it step
            if iloop > 0 and iloop <= nloop_train and int(it >= itstim):  # might be a solution
                w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, dw_buffer)
        
        # Save all data for this loop (single write per loop)
        # changed from .csv to .npy.....
        # np.savetxt(path_inputs, np.array(inputs_loop), delimiter=',')
        # np.savetxt(path_nspikes, np.array(nspikes_loop), delimiter=',')
        # np.savetxt(path_outputs, np.array(outputs_loop), delimiter=',')
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
        # seed_results.append([r1, r2, pqif, seed, iloop, sigma2, tau_rec, 
        #                     tau_div, tau_con, tau_chn])
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


# -------------------- MAIN EXECUTION --------------------
if __name__ == '__main__':

    # Seeding

    shared_rng = np.random.default_rng(MASTER_SEED)  # Set master seed
    log_path = SAVE_DIR / f"{sim_name}.out"  # Set path to output file
    config_path = save_experiment_config()  # Make the experiment.json file
    print(f"Saved experiment configuration to {config_path}")

    # Track time
    start_time = time.perf_counter()
    today = date.today()

    # Write to terminal and output file
    log_fh = open(
        log_path,
        mode="w",
        encoding="utf-8",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_fh)
    sys.stderr = Tee(original_stderr, log_fh)
    
    iout = np.linspace(0, N, num=N, endpoint=False).astype('int')



    try:  # If at some point during simulation it crashes, stdout/stderr are still restored and log file is properly closed at the end

        # Some print statements
        print("-"*60)
        print(f"EXPERIMENT: {sim_name}")
        print("-"*60)
        print(f"   date: {today}")
        print(f"   target dynamics: {target_dynamics}")
        print(f"   simulations: {len(configs)}")
        print(f"   pqif values: {pqif_values}")
        print(f"   seeds per condition: {cant_seed}")
        print(f"   parallel PQIF jobs: {parallelize_pqif}")
        print(f"   parallel seed jobs: {parallelize_seed}")
        print("-"*60)
        # -------------------- Generate target and external stimulus current --------------------
        # Generate target pattern once (shared across all simulations)
        print(f"Generating target of type {target_dynamics}...")
        if target_dynamics == "oscillations":
            (target, amp, phase, omega, romega_vec) = generate_oscillations_target( romega1=r1, romega2=r2, amp0=amp0, rng=shared_rng)
            # Save target and shared iext at the level of root folder, because it is shared across all (simulations, pqif, seed)
            save_target(
                target=target,
                phase=phase,
                omega=omega,
                romega_vec=romega_vec,
                amp=amp,
                amp0=amp0,
                r1=r1,
                r2=r2
            )
        elif target_dynamics == "sequences":
            target, neuron_permutation, gaussian_profile, sg = generate_sequences_target(sg_index, omegagauss, amp0, rng=shared_rng)
            save_sequence_target(
                target=target,
                neuron_permutation=neuron_permutation,
                gaussian_profile=gaussian_profile,
                sg=sg,
                sg_index=sg_index,
                omegagauss=omegagauss,
                amp0=amp0
            )
        else:
            print(f"No valid target dynamics was provided. Valid dynamics are 'oscillations' or 'sequences'.")
            sys.exit()  # Exits program

        print("Target generated\n")

        print("Generating external current...")
        Iext = currents(N, itmax, shared_rng)
        print("External current generated\n")


        # save_matrix_csv(
        #     Iext,
        #     SAVE_DIR / "external_current.csv",
        # )
        np.save(
            SAVE_DIR / "external_current.npy",
            Iext  # has shape (N, itmax)
        )

        # -------------------- Simulation --------------------

        # Iterate over each vt/vreset configuration
        for num_simulation, config in enumerate(configs, start=1):
            vt = config['vt']
            vreset = config['vreset']

            print(f"\n{'-'*70}")
            print(f"  STARTING SIMULATION {num_simulation}: vt={vt}, vreset={vreset}")
            print(f"{'-'*70}\n")


            # -------------------- Create folders and files for this simulation --------------------

            # folder_name, subfolder_activity, subfolder_connectivity_matrix, subfolder_currents, subfolder_inputs, subfolder_outputs, subfolder_nspikes = create_directory(num_simulation)  # might be removed

            folder_name, folders = create_directory(num_simulation)
            
            filename_results = f'simulation_{num_simulation}_results.csv'

            # delete:

            # create_parameters_file(filename_results,
            # num_simulation,
            # folder_name,
            # b,
            # vt,
            # vreset,
            # pqif_values,
            # target_dynamics,
            # r1=r1,
            # r2=r2,
            # sg_index=sg_index,
            # omegagauss=omegagauss,
            # )
            
            # Create results file with header
            csv_file_path = folder_name / filename_results
            with open(csv_file_path, 'w', newline='') as f:
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

            for pqif in pqif_values:
                save_neuron_assignment(
                    pqif=pqif,
                    folder_name=folder_name,
                )


            # -------------------- Run pqif simulation (it calls the seed function) --------------------

            # Parallelize over pqif for this simulation, let parent process write the combined result file
            parallel_results = Parallel(
                n_jobs=parallelize_pqif,
                verbose=1
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

            all_simulation_results = []

            for pqif_seed_results, pqif_currents in parallel_results:
                all_simulation_results.extend(pqif_seed_results)

            with open(csv_file_path, "a", newline="") as file_res:
                writer_res = csv.writer(file_res)
                writer_res.writerows(all_simulation_results)
            
            print(f"    SIMULATION {num_simulation} COMPLETED")
        

        print("\n" + "-"*60)
        print("ALL SIMULATIONS COMPLETED SUCCESSFULLY")
        elapsed_time = time.perf_counter() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"Total time taken: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
            f"(hh:mm:ss)"
        )
        print(f"   Output log saved to: {log_path}")
        print("-"*60)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fh.close()