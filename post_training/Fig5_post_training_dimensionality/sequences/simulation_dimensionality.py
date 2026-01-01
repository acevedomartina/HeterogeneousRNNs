import numpy as np
import pandas as pd
from scipy import sparse
import os
from numba import jit
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

idx = 1  # Número de simulación
parameters_path = f'simulacion_{idx}/simulacion_{idx}_parametros.csv'
df_params = pd.read_csv(parameters_path)

N = int(df_params.loc[0, 'N'])
N2 = int(N/2)
p = df_params.loc[0, 'p']
#b = df_params.loc[0, 'b']
b = 1/5
dt = df_params.loc[0, 'dt']
itmax = 1  # iterations to average
nloop = 1000  # number of time windows
itstim = 200  # points to discard at the beginning
sigman = df_params.loc[0, 'sigman']
v_threshold = df_params.loc[0, 'vt']
v_rest = df_params.loc[0, 'vrest']
print(v_rest)

path = os.getcwd()

iout = np.array((0, N2, N-1))
pqif_vector = [0, 0.25, 0.5, 0.75, 1]

# NUEVO: Lista de valores omegagauss a procesar
omegagauss_values = [0.5]


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


def simulate_single_condition(omegagauss, pqif, vt, vrest, seed, N, p, b, dt, itmax, 
                              nloop, Iext, sigman, outdir, idx, itstim, plot_verification=False):
    """Simulates a single complete condition for a specific omegagauss"""
    
    # Unique seed for this simulation
    np.random.seed(seed)
    
    nqif = int(N * pqif)
    x = np.random.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    
    # MODIFICADO: Cargar pesos desde la carpeta omega específica
    carpeta_omega = f'simulacion_{idx}/omega_{omegagauss}'
    path_pesos = f'{carpeta_omega}/matrices_pesos/pesos_pqif_{pqif}_matriz_iloop_11_semilla_{seed}'
    
    if not os.path.exists(path_pesos):
        return f"✗ Error: No se encontró {path_pesos}"
    
    df_w = pd.read_csv(path_pesos, header=None)
    w = df_w.values
    
    # MODIFICADO: Cargar targets desde la carpeta omega específica
    targets_path = f'{carpeta_omega}/targets_pqif_{pqif}.csv'
    
    if not os.path.exists(targets_path):
        return f"✗ Error: No se encontró {targets_path}"
    
    targets = pd.read_csv(targets_path, header=0).values
    
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
    
    # Verification plot (opcional)
    if plot_verification:
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, rprom[5, :], label='rprom[5] (simulated)', alpha=0.7)
        plt.plot(targets[:, 5], label='target[5]', alpha=0.7, linestyle='--')
        plt.xlabel('iloop')
        plt.ylabel('r')
        plt.legend()
        plt.title(f'Neuron 5: omega={omegagauss}, pqif={pqif}, seed={seed}')
        plt.show()
    
    
    # MODIFICADO: Guardar con omegagauss en el nombre
    vrest_str = "None" if vrest is None else str(vrest)
    vt_str = "None" if vt is None else str(vt)
    fname = f"rprom_omega_{omegagauss}_pqif_{pqif}_vt_{vt_str}_vrest_{vrest_str}_seed_{seed}.npy"
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

# Build list of all conditions (MODIFICADO: incluye omegagauss)
all_conditions = []

for omegagauss in omegagauss_values:
    carpeta_omega = f'simulacion_{idx}/omega_{omegagauss}'
    
    for pqif in pqif_vector:
        # MODIFICADO: Cargar Iext desde la carpeta omega específica
        Iext_path = f'{carpeta_omega}/Iext_pqif_{pqif}_seed_0.csv'
        
        if not os.path.exists(Iext_path):
            print(f"⚠️  Advertencia: No se encontró {Iext_path}, saltando...")
            continue
        
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
                    all_conditions.append((omegagauss, pqif, vt, vrest, seed, Iext))

print(f"📊 Total simulations: {len(all_conditions)}")
print(f"📊 Omega values: {omegagauss_values}")
print(f"📊 pqif values: {pqif_vector}")

# Run simulations
print("\n" + "="*60)
print("Running simulations...")
print("="*60 + "\n")

# Option 1: WITHOUT parallelization (to see plots)
# Uncomment to run without parallelization and see verification plots
"""
for omegagauss, pqif, vt, vrest, seed, Iext in all_conditions:
    result = simulate_single_condition(
        omegagauss, pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, 
        Iext, sigman, outdir, idx, itstim, plot_verification=True
    )
    print(result)
"""

# Option 2: WITH parallelization (faster, no plots)
all_results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_single_condition)(
        omegagauss, pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, 
        Iext, sigman, outdir, idx, itstim, plot_verification=True
    )
    for omegagauss, pqif, vt, vrest, seed, Iext in all_conditions
)

# Print results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
for result in all_results:
    print(result)

print("\n✅ All simulations completed!")
print(f"📁 Results saved in: {outdir}/")