import numpy as np
import pandas as pd
from scipy import sparse
import os
from numba import jit
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

########## Make correct path ##########

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent # Parent of cwd


idx = 4
parameters_path = BASE_DIR / f'simulation_{idx}/simulation_{idx}_parameters.csv'
# learning\oscillations\simulation_2\simulation_2_parameters.csv
df_params = pd.read_csv(parameters_path)

########## TEMPORARY FIX: I make it go to the correct working directory so all paths work for now instead of using BASE_DIR

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # current working directory becomes folder containing the script


N = int(df_params.loc[0, 'N'])
N2 = int(N/2)
p = df_params.loc[0, 'p']
b = df_params.loc[0, 'b']
dt = df_params.loc[0, 'dt']
itmax = 50 # iterations to average
nloop = 2000  # number of time windows
itstim = 200  # points to discard at the beginning
sigman = df_params.loc[0, 'sigman']
v_threshold = df_params.loc[0, 'vt']
v_rest = df_params.loc[0, 'vrest']

path = os.getcwd()

iout = np.array((0, N2, N-1))
pqif_vector = [0, 0.25,  0.5, 0.75, 1]

gsyn_vect = [0]

outdir = f'rprom/simulacion_{idx}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

############ OPTIMIZED FUNCTIONS ##############

@jit(nopython=True)
def dynamics_jit(x_var, r_var, I_var, nqif, b, N):
    """JIT-compiled dynamics for speed"""
    dx = np.zeros(N)
    
    # LIF
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:]
    
    # QIF
    cos_x = np.cos(x_var[:nqif])
    dx[:nqif] = 1 - cos_x + I_var[:nqif] * (1 + cos_x)
    
    dr = -b * r_var
    return dx, dr


@jit(nopython=True)
def detect_jit(x, xnew, rnew, nspike, nqif, b, vt, vrest, N):
    """JIT-compiled spike detection"""
    # LIF
    for i in range(nqif, N):
        if x[i] < vt and xnew[i] > vt:
            rnew[i] += b
            xnew[i] = vrest
            nspike[i] += 1
    
    # QIF
    for i in range(nqif):
        dpi = np.pi - np.mod(x[i], 2*np.pi)
        if dpi < 0:
            dpi += 2*np.pi
        
        dx = xnew[i] - x[i]
        if dx > 0 and (dx - dpi) > 0:
            rnew[i] += b
            nspike[i] += 1
    
    return xnew, rnew, nspike


def evolution_optimized(x, r, w, nqif, II, sigman, dt, b, vt, vrest, N):
    """Optimized evolution step"""
    I_noise = np.random.randn(N) * sigman
    v = w.dot(r)
    I_total = II + v + I_noise
    
    # RK2 integration
    dx, dr = dynamics_jit(x, r, I_total, nqif, b, N)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    
    dx, dr = dynamics_jit(xnew, rnew, I_total, nqif, b, N)
    xnew = x + dt * dx
    rnew = r + dt * dr
    
    # Spike detection
    nspike = np.zeros(N)
    xnew, rnew, nspike = detect_jit(x, xnew, rnew, nspike, nqif, b, vt, vrest, N)
    
    return xnew, rnew, nspike


def simulate_single_condition(pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir, idx, itstim):
    """Simulates a single complete condition"""
    
    # Unique seed for this simulation
    np.random.seed(seed)
    
    nqif = int(N * pqif)
    x = np.random.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    
    # Connectivity weights
    path_pesos = f'simulation_{idx}/simulation_{idx}_connectivity_matrix/simulation_{idx}_connectivity_pqif_{pqif}_iloop_11_seed_0'
    df_w = pd.read_csv(path_pesos, header=None)
    w = df_w.values
    
    targets = pd.read_csv(f'simulation_{idx}/simulation_{idx}_targets_{pqif}.csv', header=0).values
    
    # Pre-allocate arrays
    rprom = np.zeros((N, nloop), dtype=np.float32)
    nspike_total = np.zeros(N)
    time_vector = np.arange(nloop) * itmax
       
    # System evolution
    for iloop in range(nloop):
        r_sum = np.zeros(N, dtype=np.float32)
        
        # Average over itmax iterations
        for it in range(itmax):
            # Global time index: each window iloop has itmax steps
            time_idx = iloop * itmax + it
            
            
            # Get Iext corresponding to this time step
            if time_idx < Iext.shape[1]:
                II = Iext[:, time_idx]
            else:
                II = np.zeros(N)  # If we exceed the size, use zeros
            
            x, r, nspike = evolution_optimized(
                x, r, w, nqif, II, sigman, dt, b,
                vt if vt is not None else 0,
                vrest if vrest is not None else 0,
                N
            )
            

            r_sum += r
            nspike_total += nspike
        
        # Average over itmax iterations
        rprom[:, iloop] = r_sum / itmax
    
    # Verification plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_vector, rprom[5, :], label='rprom[7] (simulated)', alpha=0.7)
    plt.plot(targets[:, 5], label='target[7]', alpha=0.7, linestyle='--')
    #plt.plot(targets[:, 7], label='target[7]', alpha=0.7, linestyle='--')
    plt.xlabel('iloop')
    plt.ylabel('r')
    plt.legend()
    plt.title(f'Neuron 7: pqif={pqif}, seed={seed}')
    plt.show()
    
    # Discard the first itstim points
    rprom_cut = rprom[:, itstim:]
    
    # Compute covariance (transpose to have variables in rows)
    #ccorr = np.cov(rprom_cut, bias=False, rowvar=True)
    
    # Save results
    vrest_str = "None" if vrest is None else str(vrest)
    vt_str = "None" if vt is None else str(vt)
    fname = f"rprom_pqif_{pqif}_vt_{vt_str}_vrest_{vrest_str}_seed_{seed}.npy"
    fpath = os.path.join(outdir, fname)

    if np.isfinite(rprom).all():
        np.save(fpath, rprom)
        return f"✓ Saved: {fname}"
    else:
        return f"✗ Warning: rprom contains NaN/Inf for {fname}"


########## PARALLEL SIMULATION ##############

# Number of available CPUs
n_cores = multiprocessing.cpu_count()
print(f"🚀 Using {n_cores} CPU cores for parallelization")
print(f"📊 Total simulations to compute...")
print("-" * 60)

# Build list of all conditions
all_conditions = []
for pqif in pqif_vector:
    Iext_path = f'simulation_{idx}/simulation_{idx}_Iext_pqif_{pqif}_seed_0.csv'
    Iext = pd.read_csv(Iext_path, header=None).values
    
    if pqif == 1:
        vt_vect = [None]
    else:
        vt_vect = [0]

    for vt in vt_vect:
        if vt == 0:
            vrest_vect = [v_rest]
        elif vt == 0.5:
            vrest_vect = [-1, -3.1, -8.5]
        else:
            vrest_vect = [None]

        for vrest in vrest_vect:
            for seed in range(1):
                all_conditions.append((pqif, vt, vrest, seed, Iext))

print(f"📊 Total simulations: {len(all_conditions)}")

# Run simulations (comment out Parallel if you want to see the plots)
print("Running simulation...")

# Option 1: WITHOUT parallelization (to see plots)
for pqif, vt, vrest, seed, Iext in all_conditions:
    result = simulate_single_condition(
        pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir, idx, itstim
    )
    print(result)

# Option 2: WITH parallelization (comment out the plots inside the function)
# all_results = Parallel(n_jobs=-1, verbose=10)(
#     delayed(simulate_single_condition)(
#         pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir, idx, itstim
#     )
#     for pqif, vt, vrest, seed, Iext in all_conditions
# )