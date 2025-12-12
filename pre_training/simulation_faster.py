import numpy as np
import pandas as pd
from scipy import sparse
import os
from numba import jit
from joblib import Parallel, delayed
import multiprocessing

# Parámetros de simulación
T = 1000
dt = 0.1
sigman = 0.0
v_threshold = 0
v_rest = [-22, -12.3, -17, -8.5]
path = os.getcwd()

N = 200
N2 = int(N/2)
p = 0.3
b = 0.5
dt = 0.1
itmax = 50
nloop = 2000
sigman = 1

iout = np.array((0, N2, N-1))
pqif_vector = [0, 1]

gsyn_vect =[0]
Iext = np.zeros((N, itmax))

outdir = 'cov_matrices'
if not os.path.exists(outdir):
    os.makedirs(outdir)

############ FUNCIONES OPTIMIZADAS ##############

def initialize_weights(N, p, gsyn, seed=None):
    """Optimized weight initialization"""
    if seed is not None:
        np.random.seed(seed)
    
    w = sparse.random(N, N, p, data_rvs=np.random.randn, format='csr')
    w.setdiag(0)
    w = w.tocsr()
    
    # Row mean centering
    w_dense = w.toarray()
    w_dense *= (gsyn / np.sqrt(p * N))
    for i in range(N):
        row = w_dense[i, :]
        nonzero_idx = np.nonzero(row)[0]
        if len(nonzero_idx) > 0:
            av0 = np.mean(row[nonzero_idx])
            row[nonzero_idx] -= av0
    
    return sparse.csr_matrix(w_dense)


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


def evolution_optimized(x, r, w_csr, nqif, II, sigman, dt, b, vt, vrest, N):
    """Optimized evolution step"""
    I_noise = np.random.randn(N) * sigman
    v = w_csr.dot(r)
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


def simulate_single_gsyn(gsyn, pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir):
    """Simula un único valor de gsyn - esta función se paralelizará"""
    
    # Seed único para esta simulación
    sim_seed = seed * 10000 + int(gsyn * 100)
    np.random.seed(sim_seed)
    
    nqif = int(N * pqif)
    x = np.random.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    
    w_csr = initialize_weights(N, p, gsyn, seed=sim_seed)
    
    # Pre-allocate arrays
    rprom = np.zeros((N, nloop), dtype=np.float32)
    nspike_total = np.zeros(N)

    for iloop in range(nloop):
        r_sum = np.zeros(N, dtype=np.float32)
        
        for it in range(itmax):
            II = Iext[:, it]
            x, r, nspike = evolution_optimized(
                x, r, w_csr, nqif, II, sigman, dt, b,
                vt if vt is not None else 0,
                vrest if vrest is not None else 0,
                N
            )
            r_sum += r
            nspike_total += nspike
        
        rprom[:, iloop] = r_sum / itmax

    # Compute covariance
    ccorr = np.cov(rprom, bias=False, rowvar=True)

    # Save results
    vrest_str = "None" if vrest is None else str(vrest)
    vt_str = "None" if vt is None else str(vt)
    fname = f"FASTER_cov_pqif_{pqif}_vt_{vt_str}_vrest_{vrest_str}_gsyn_{gsyn:.2f}_seed_{seed}.npy"
    fpath = os.path.join(outdir, fname)

    if np.isfinite(ccorr).all():
        np.save(fpath, ccorr)
        return f"✓ Guardado: {fname}"
    else:
        return f"✗ Advertencia: ccorr contiene NaN/Inf para {fname}"


def simulate_condition(pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir, gsyn_vect):
    """Simula una condición completa (todos los gsyn) - nivel medio de paralelización"""
    
    print(f"Iniciando: pqif={pqif}, vt={vt}, vrest={vrest}, seed={seed}")
    
    # Paralelizar sobre gsyn_vect
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_single_gsyn)(
            gsyn, pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir
        )
        for gsyn in gsyn_vect
    )
    
    print(f"Completado: pqif={pqif}, vt={vt}, vrest={vrest}, seed={seed}")
    return results


########## SIMULACION PARALELA ##############

# Número de CPUs disponibles
n_cores = multiprocessing.cpu_count()
print(f"🚀 Usando {n_cores} núcleos de CPU para paralelización")
print(f"📊 Total de simulaciones: {len(pqif_vector) * len(v_rest) * len(gsyn_vect)}")
print("-" * 60)

# Construir lista de todas las condiciones
all_conditions = []
for pqif in pqif_vector:
    if pqif == 1:
        vt_vect = [None]
    else:
        vt_vect = [0]

    for vt in vt_vect:
        if vt == 0:
            vrest_vect = v_rest
        elif vt == 0.5:
            vrest_vect = [-1, -3.1, -8.5]
        else:
            vrest_vect = [None]

        for vrest in vrest_vect:
            for seed in range(20):
                all_conditions.append((pqif, vt, vrest, seed))

# OPCIÓN 1: Paralelización a nivel de condición (recomendada para muchas condiciones)
print("Ejecutando simulación paralela...")
all_results = []
for pqif, vt, vrest, seed in all_conditions:
    results = simulate_condition(
        pqif, vt, vrest, seed, N, p, b, dt, itmax, nloop, Iext, sigman, outdir, gsyn_vect
    )
    all_results.extend(results)

print("\n" + "="*60)
print("🎉 ¡Simulación completada!")
print(f"📁 Archivos guardados en: {outdir}/")
print("="*60)