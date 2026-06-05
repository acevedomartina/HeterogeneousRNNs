# ========== Oscillations (parallelized) ==========

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os
from joblib import Parallel, delayed

####### Global parameters #######
# Neurons
N = 200                 # Number of nodes (neurons)
N2 = int(N/2)           # Half

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
amp_corriente = 20      # Stimulus intensity
amp0 = 4                # Used in target. Changed from 8 to 4, in order to have the same current amplitudes as in the pre-training case for both oscillations and sequences
# Training
nloop = 16              # Number of loops, 0: pre-training, last: post-training
nloop_train = 10        # Last training loop
cant_seed = 50          # Independent simulations
ts = 5                  # 
b = 1 / ts              # adaptation parameter for r - In evolution of r, dr/dt = -b * r, b is magnitude
ftrain = 1              # Fraction of neurons to train

####### File organization functions #######

def crear_subcarpeta(carpeta_padre, nombre_subcarpeta):
    '''
    Joins path components (folder and subfolder) if not existing already

    carpeta_padre: Parent folder name
    nombre_subcarpeta: Subfolder name
    '''
    ruta = os.path.join(carpeta_padre, nombre_subcarpeta)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    return ruta

def crear_carpetas(num_simulacion): 
    '''
    Main simulation folder with connectivity matrix subfolder only
    '''
    nombre_carpeta = f"simulation_{num_simulacion}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

    # Only create the connectivity matrix subfolder
    sub_pesos = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_connectivity_matrix")

    return nombre_carpeta, sub_pesos

def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta, b, vt, vrest):
    '''    
    Save simulation parameters to file
    '''
    data_parametros = {
        'N': [N],
        'p': [p],
        'gsyn': [gsyn],
        'nloop': [nloop],
        'nloop_train':[nloop_train],
        'cant_seed': [cant_seed],
        'dt': [dt],
        'itmax': [itmax],
        'itstim': [itstim],
        'amp_corriente': [amp_corriente],
        'amp0': [amp0],
        'ftrain': [ftrain],
        'alpha': [alpha],
        'sigman': [sigman],
        'vt': [vt],
        'b': [b],
        'vrest': [vrest],
        'results_file': [filename_resultados],
    }

    df = pd.DataFrame(data_parametros)
    filename_parametros = f'simulation_{num_simulacion}_parameters.csv'
    csv_parametros_path = os.path.join(nombre_carpeta, filename_parametros)
    df.to_csv(csv_parametros_path, index=False)

####### Function to generate target patterns #######

def generate_target(romega1, romega2, amp0):
    '''
    Generates oscillatory target with theta and gamma (to replicate oscillatory frequencies in the MEC)
    ----------
    romega1: theta
    romega2: gamma (5*theta)
    amp0: 
    '''
    target=np.zeros((N,itmax))  # Neuron (row) activity over timesteps (columns)
    amp=np.random.uniform(size=N)*amp0
    phase=np.random.uniform(0,2*np.pi,size=N)
    indices = [i for i in range(N)]
    indices = np.random.permutation(indices) # Indices to identify which neuron is assigned each frequency
    
    romega_vec = np.zeros(N)  # (N,)
    
    for i in range(N2):
        # Assigning frequencies to indices
        romega_vec[indices[i]]= romega1
        romega_vec[indices[i+N2]]=romega2
    
    omega=romega_vec*2*np.pi/itmax

    for it in range(itmax):
        target[:,it]=amp*np.cos(it*omega+phase) 
            
    return target, amp, phase, omega, romega_vec, amp0


def guardar_matriz_csv(matriz, nombre_archivo):
    ''' 
    Save matrix to CSV
    '''
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for fila in matriz:
            fila_lista = [str(elemento) for elemento in fila.flat]  # rows
            escritor_csv.writerow(fila_lista)


####### Dynamics and learning functions #######

def dynamics(x_var, r_var, I_var, nqif, b):
    '''
    Dynamics of neurons.
    Compute derivatives of neuron state and adaptation variable.
    
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
    dx = np.zeros(N)

    # LIF neurons noise
    I_noise_lif = np.random.randn(N - nqif) * sigman
    # QIF neurons noise
    I_noise_qif = np.random.randn(nqif) * sigman

    # LIF dynamics: dx/dt = -x + I + noise
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif

    # QIF dynamics: dx/dt = 1 - cos(x) + I*(1 + cos(x)) + noise
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif])) + I_noise_qif

    # Adaptation: dr/dt = -b * r
    dr = -b * r_var

    return dx, dr


def detect(x, xnew, rnew, nspike, nqif, b, vt, vrest):
    # LIF spike detection
    ispike_lif = np.where(x[nqif:] < vt) and np.where(xnew[nqif:] > vt)
    ispike_lif = ispike_lif[0] + nqif
    if len(ispike_lif) > 0:
        rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
        xnew[ispike_lif[:]] = vrest
        nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
    # QIF spike detection
    dpi = np.mod(np.pi - np.mod(x, 2*np.pi), 2*np.pi)
    ispike_qif = np.where((xnew[:nqif]-x[:nqif]) > 0) and np.where((xnew[:nqif]-x[:nqif]-dpi[:nqif]) > 0)
    if len(ispike_qif) > 0:
        rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
        nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
    return xnew, rnew, nspike


def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vrest):
    II = np.squeeze(np.asarray(Iext[:, it]))
    v = w.dot(r.T).A1
    dx, dr = dynamics(x, r, II + v, nqif, b)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr = dynamics(xnew, rnew, II + v, nqif, b)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vrest)
    x, r = np.copy(xnew), np.copy(rnew)
    return x, r, nspike, r[iout], II, v


def initialize_connectivity_matrix(N, p, gsyn):
    w = sparse.random(N, N, p, data_rvs=np.random.randn).todense()
    np.fill_diagonal(w, 0)  # No autapses
    w *= gsyn / np.sqrt(p * N)
    
    for i in range(N):
        i0 = np.where(w[i, :])[1]
        if len(i0) > 0:
            av0 = np.sum(w[i, i0]) / len(i0)
            w[i, i0] -= av0  # Subtract mean so each row has zero mean
    
    return w


def initialize_neurons(N):
    x = np.random.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    nspike = np.zeros(N)
    return x, r, nspike


def initialize_training(N, w):
    nind = np.zeros(N).astype('int')
    idx = []
    P = []
    for i in range(N):
        ind = np.where(w[i, :])[1]
        nind[i] = len(ind)
        idx.append(ind)
        P.append(np.identity(nind[i]) / alpha)
    return P, idx


def currents(N, itmax):
    Iext = np.zeros((N, itmax))
    Ibac = amp_corriente * (2 * np.random.uniform(size=N) - 1)
    Iext[:, :itstim] = Ibac[:, None]
    return Iext


def learning(it, iloop, w, r, P, idx, target, norm_w0):
    error = target[:, it:it + 1] - w @ r.reshape(N, 1)
    for i in range(N):
        ri = r[idx[i]].reshape(len(idx[i]), 1)
        k1 = P[i] @ ri
        k2 = ri.T @ P[i]
        den = 1 + ri.T @ k1
        P[i] -= (k1 @ k2) / den
        dw = error[i, 0] * P[i] @ r[idx[i]]
        w[i, idx[i]] += dw

    return w, P


####### Motifs and dimensionality calculations #######

def motifs(w, gsyn, N):
    w = w - np.mean(w)
    
    ww = np.matmul(w, w)
    wtw = np.matmul(w.T, w)
    wwt = np.matmul(w, w.T)
    
    sigma2 = np.trace(wwt) / N
    
    tau_rec = np.trace(ww)
    tau_rec /= sigma2 * N
    
    tau_div = np.sum(wwt) - np.trace(wwt)
    tau_div /= sigma2 * N * (N-1)
    
    tau_con = np.sum(wtw) - np.trace(wtw)
    tau_con /= sigma2 * N * (N-1)
    
    tau_chn = 2 * (np.sum(ww) - np.trace(ww))
    tau_chn /= sigma2 * N * (N-1)
    
    return sigma2, tau_rec, tau_div, tau_con, tau_chn


####### Parallelized simulation functions #######

def run_single_seed(seed, pqif, num_simulacion, vt, vrest,
                    N, N2, p, gsyn, nloop, nloop_train,
                    dt, itmax, itstim, amp_corriente, alpha, sigman,
                    b, iout, nombre_carpeta, sub_pesos):
    """
    Run complete simulation for a single seed.
    Each seed generates its own target pattern using its seed value.
    Only connectivity matrices are saved to disk.
    """

    nqif = int(N * pqif)

    # Fix seed before generating target and initializing everything
    np.random.seed(seed=seed)

    # Generate a unique target for this seed
    target, amp, phase, omega, romega_vec, amp0_local = generate_target(
        romega1=1, romega2=5, amp0=amp0
    )

    # Initialize network
    x, r, nspike = initialize_neurons(N)
    Iext = currents(N, itmax)

    # Initialize connectivity
    w = initialize_connectivity_matrix(N, p, gsyn)
    norm_w0 = np.linalg.norm(w)
    P, idx = initialize_training(N, w)

    # Storage for results across all loops
    seed_results = []

    # Main training loop
    for iloop in range(nloop):
        print(f"Simulation {num_simulacion} - Seed {seed} - pqif {pqif} - Loop {iloop}/{nloop-1}")

        # Time evolution for this loop
        for it in range(itmax):
            nspike = np.zeros(N)

            x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt,
                                                  iout, nspike, b, vt=vt, vrest=vrest)

            # Apply learning rule during training period
            if iloop > 0 and iloop <= nloop_train and int(it > itstim):
                w, P = learning(it, iloop, w, r, P, idx, target, norm_w0)

        # Calculate network motifs
        sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)

        # Save weight matrix at specific loops only
        if iloop == 0 or iloop == (nloop_train + 1):
            path_w_seed = os.path.join(sub_pesos,
                                       f'simulation_{num_simulacion}_connectivity_pqif_{pqif}_iloop_{iloop}_seed_{seed}')
            guardar_matriz_csv(w, path_w_seed)

        # Store results for this loop
        seed_results.append([pqif, seed, iloop, sigma2, tau_rec,
                              tau_div, tau_con, tau_chn])

    return seed_results


def run_pqif_simulation(pqif, num_simulacion, vt, vrest,
                        N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                        dt, itmax, itstim, amp_corriente, alpha, sigman,
                        b, iout):
    """
    Run simulation for a specific pqif value, parallelizing over seeds.
    Each seed independently generates its own target inside run_single_seed.
    Only connectivity matrices are saved to disk.
    """

    print(f"\n{'='*60}")
    print(f"Simulation {num_simulacion} - Processing pqif = {pqif}")
    print(f"vt={vt}, vrest={vrest}")
    print(f"Parallelizing over {cant_seed} seeds")
    print(f"{'='*60}\n")

    nombre_carpeta = f"simulation_{num_simulacion}"
    sub_pesos = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_connectivity_matrix")

    filename_resultados = f'simulation_{num_simulacion}_results.csv'
    csv_file_path = os.path.join(nombre_carpeta, filename_res...