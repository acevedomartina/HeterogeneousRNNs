import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os
from joblib import Parallel, delayed

####### Global parameters #######
N = 200
N2 = int(N/2)
p = 0.3
gsyn = 0.5
alpha = 0.25
dt = 0.1
itmax = 1000
sigman = 1
itstim = 200
amp_corriente = 20
amp0 = 4
nloop = 16
nloop_train = 10
cant_seed = 10
ts = 5
b = 1 / ts
ftrain = 1

####### File organization functions #######
def crear_subcarpeta(nombre_carpeta, nombre_subcarpeta):
    subcarpeta_path_total = (os.path.join(nombre_carpeta, nombre_subcarpeta))
    if not os.path.exists(subcarpeta_path_total):
        os.makedirs(subcarpeta_path_total)
    return subcarpeta_path_total


def crear_subcarpeta(carpeta_padre, nombre_subcarpeta):
    ruta = os.path.join(carpeta_padre, nombre_subcarpeta)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    return ruta

def crear_carpetas(num_simulacion): 
    # Main simulation folder
    nombre_carpeta = f"simulation_{num_simulacion}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

    # Subfolders within the simulation
    sub_act = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_activity_examples")
    sub_pesos = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_connectivity_matrix")
    sub_corrientes = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_currents")
    sub_inputs = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_inputs")
    sub_outputs = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_outputs")
    sub_nspikes = crear_subcarpeta(nombre_carpeta, f"simulation_{num_simulacion}_nspikes")

    return nombre_carpeta, sub_act, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes


def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta, b, vt, vrest):
    # Save simulation parameters to file
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
    target=np.zeros((N,itmax))
    amp=np.random.uniform(size=N)*amp0
    phase=np.random.uniform(0,2*np.pi,size=N)
    indices = [i for i in range(N)]
    indices = np.random.permutation(indices) # Indices to identify which neuron is assigned each frequency
    
    romega_vec = np.zeros(N)
    
    for i in range(N2):
        romega_vec[indices[i]]= romega1
        romega_vec[indices[i+N2]]=romega2
    
    omega=romega_vec*2*np.pi/itmax

    for it in range(itmax):
        target[:,it]=amp*np.cos(it*omega+phase) 
            
    return target, amp, phase, omega, romega_vec, amp0


def save_target(target, phase, omega, romega_vec, amp, amp0, num_simulacion, nombre_carpeta, pqif):
    # Save target parameters and values to CSV
    data = {'Neurona': range(N), 'Fase': phase, 'Frecuencia': omega, 'romega': romega_vec, 'Amplitud': amp, 'amp0': amp0}
    df = pd.DataFrame(data)
    nombre_archivo = f'simulation_{num_simulacion}_targets_parameters.csv'
    csv_target_path = os.path.join(nombre_carpeta, nombre_archivo)
    df.to_csv(csv_target_path, index=False)

    target_df = pd.DataFrame(target.T, columns=[f'Neurona_{i}' for i in range(N)])
    nombre_archivo_target = f'simulation_{num_simulacion}_targets_{pqif}.csv'
    csv_target_path = os.path.join(nombre_carpeta, nombre_archivo_target)
    target_df.to_csv(csv_target_path, index=False)


def guardar_matriz_csv(matriz, nombre_archivo):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for fila in matriz:
            fila_lista = [str(elemento) for elemento in fila.flat]
            escritor_csv.writerow(fila_lista)


####### Dynamics and learning functions #######

def dynamics(x_var,r_var,I_var,nqif, b):
    dx=np.zeros(N)
    I_noise_lif = np.random.randn(N - nqif)*sigman 
    I_noise_qif = np.random.randn(nqif)*sigman
    # LIF neurons
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif
    # QIF neurons
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif])) + I_noise_qif
    dr = -b*r_var
    return dx,dr


def detect(x,xnew,rnew,nspike,nqif, b, vt, vrest):
    # LIF spike detection
    ispike_lif=np.where(x[nqif:]<vt) and np.where(xnew[nqif:]>vt)
    ispike_lif=ispike_lif[0]+nqif
    if(len(ispike_lif)>0):
        rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
        xnew[ispike_lif[:]] = vrest
        nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
    # QIF spike detection
    dpi=np.mod(np.pi - np.mod(x,2*np.pi),2*np.pi)  # distance to pi
    ispike_qif=np.where((xnew[:nqif]-x[:nqif])>0) and np.where((xnew[:nqif]-x[:nqif]-dpi[:nqif])>0)
    if(len(ispike_qif)>0):
        rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
        nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
    return xnew,rnew,nspike

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
            w[i, i0] -= av0
    
    return w

def initialize_neurons(N):
    x = np.random.uniform(size=N) * 2 * np.pi
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

def currents(N, itmax):
    Iext=np.zeros((N,itmax))
    Ibac=amp_corriente*(2*np.random.uniform(size=N)-1)
    Iext[:, :itstim] = Ibac[:, None]  # Vectorized assignment
    return Iext


def learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer):
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
        csv_writer.writerow([modt_value, modw_value])
        
    return w, P


####### Motifs and dimensionality calculations #######
            
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


####### Parallelized simulation functions #######

def run_single_seed(seed, pqif, num_simulacion, vt, vrest, target, 
                    N, N2, p, gsyn, nloop, nloop_train,
                    dt, itmax, itstim, amp_corriente, alpha, sigman,
                    b, iout, nombre_carpeta, sub_pesos, sub_corrientes, 
                    sub_inputs, sub_outputs, sub_nspikes):
    """
    Run complete simulation for a single seed
    This function is parallelized over seeds
    """
    
    # Calculate nqif based on proportion of QIF neurons
    nqif = int(N * pqif)
    
    np.random.seed(seed=seed)
    
    # Initialize network
    x, r, nspike = initialize_neurons(N)
    Iext = currents(N, itmax)
    
    # Save external current
    path_Iext = os.path.join(nombre_carpeta, 
                            f'simulation_{num_simulacion}_Iext_pqif_{pqif}_seed_{seed}.csv')
    guardar_matriz_csv(Iext, path_Iext)
    
    # Initialize connectivity
    w = initialize_connectivity_matrix(N, p, gsyn)
    norm_w0 = np.linalg.norm(w)
    P, idx = initialize_training(N, w)
    
    # Prepare file for weight evolution tracking
    filename_dw = os.path.join(nombre_carpeta, 
                              f'simulation_{num_simulacion}_dw_pqif_{pqif}_seed_{seed}.csv')
    
    # Storage for results across all loops
    seed_results = []
    corrientes_buffer = []
    
    with open(filename_dw, mode='w', newline='') as file_dw:
        csv_writer_dw = csv.writer(file_dw)
        csv_writer_dw.writerow(['modt', 'modw'])
        
        # Main training loop
        for iloop in range(nloop):
            
            # Pre-allocate arrays for this loop
            outputs_loop = []
            inputs_loop = []
            nspikes_loop = []
            
            # Define output paths
            path_inputs = os.path.join(sub_inputs, 
                                      f'simulation_{num_simulacion}_inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
            path_nspikes = os.path.join(sub_nspikes, 
                                       f'simulation_{num_simulacion}_nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
            path_outputs = os.path.join(sub_outputs, 
                                       f'simulation_{num_simulacion}_outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
            
            # Time evolution for this loop
            for it in range(itmax):
                nspike = np.zeros(N)
                
                x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt, 
                                                     iout, nspike, b, vt=vt, vrest=vrest)
                
                entrada = II + v
                
                # Accumulate data in memory (more efficient than writing each iteration)
                outputs_loop.append(rout)
                inputs_loop.append(entrada)
                nspikes_loop.append(nspike)
                
                # Record currents at specific time points in specific loops
                if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                    corrientes_buffer.append([pqif, seed, iloop, it, 
                                            II[0], v[0], II[1], v[1], 
                                            II[N2+1], v[N2+1], II[N2+2], v[N2+2]])
                
                # Apply learning rule during training period
                if iloop > 0 and iloop <= nloop_train and int(it > itstim):
                    w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer_dw)
            
            # Save all data for this loop (single write per loop)
            np.savetxt(path_inputs, np.array(inputs_loop), delimiter=',')
            np.savetxt(path_nspikes, np.array(nspikes_loop), delimiter=',')
            np.savetxt(path_outputs, np.array(outputs_loop), delimiter=',')
            
            # Calculate network motifs
            sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)
            
            # Save weight matrix at specific loops
            if iloop == 0 or iloop == (nloop_train + 1):
                path_w_seed = os.path.join(sub_pesos, 
                                          f'simulation_{num_simulacion}_connectivity_pqif_{pqif}_iloop_{iloop}_seed_{seed}')
                guardar_matriz_csv(w, path_w_seed)
            
            # Store results for this loop
            seed_results.append([pqif, seed, iloop, sigma2, tau_rec, 
                               tau_div, tau_con, tau_chn])
    
    return seed_results, corrientes_buffer


def run_pqif_simulation(pqif, num_simulacion, vt, vrest, target, phase, amp, omega, romega_vec, amp0, 
                        N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                        dt, itmax, itstim, amp_corriente, alpha, sigman,
                        b, iout):
    """
    Run simulation for specific pqif value, parallelizing over seeds
    """
    
    print(f"\n{'='*60}")
    print(f"Simulation {num_simulacion} - Processing pqif = {pqif}")
    print(f"vt={vt}, vrest={vrest}")
    print(f"Parallelizing over {cant_seed} seeds")
    print(f"{'='*60}\n")
    
    # Get folder paths (already created)
    nombre_carpeta = f"simulation_{num_simulacion}"
    sub_act = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_activity_examples")
    sub_pesos = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_connectivity_matrix")
    sub_corrientes = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_currents")
    sub_inputs = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_inputs")
    sub_outputs = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_outputs")
    sub_nspikes = os.path.join(nombre_carpeta, f"simulation_{num_simulacion}_nspikes")
    
    # Results file path
    filename_resultados = f'simulation_{num_simulacion}_results.csv'
    csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
    
    # Prepare currents file header
    path_currents_seed = os.path.join(sub_corrientes, 
                                       f'simulation_{num_simulacion}_currents_pqif_{pqif}.csv')
    with open(path_currents_seed, mode='w', newline='') as file_:
        writer_ = csv.writer(file_)
        writer_.writerow(['pqif', 'seed', 'iloop', 'it', 'II_0', 'v_0', 
                        'II_1', 'v_1', 'II_N2+1', 'v_N2+1', 'II_N2+2', 'v_N2+2'])
    
    # Parallelize over seeds
    results = Parallel(n_jobs=cant_seed, verbose=5)(
        delayed(run_single_seed)(
            seed, pqif, num_simulacion, vt, vrest, target,
            N, N2, p, gsyn, nloop, nloop_train,
            dt, itmax, itstim, amp_corriente, alpha, sigman,
            b, iout, nombre_carpeta, sub_pesos, sub_corrientes,
            sub_inputs, sub_outputs, sub_nspikes
        )
        for seed in range(cant_seed)
    )
    
    # Consolidate results from all seeds
    # results is a list of (seed_results, corrientes_buffer) tuples
    all_seed_results = []
    all_corrientes = []
    for seed_results, corrientes_buffer in results:
        all_seed_results.extend(seed_results)
        all_corrientes.extend(corrientes_buffer)
    
    # Write all results to file (append mode for this pqif)
    with open(csv_file_path, 'a', newline='') as file_res:
        writer_res = csv.writer(file_res)
        writer_res.writerows(all_seed_results)
    
    # Write all currents to file
    if all_corrientes:
        with open(path_currents_seed, 'a', newline='') as f_corr:
            writer_corr = csv.writer(f_corr)
            writer_corr.writerows(all_corrientes)
    
    print(f"✓ pqif={pqif} completed for simulation {num_simulacion}\n")
    
    return num_simulacion


# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    
    iout = np.linspace(0, N, num=N, endpoint=False).astype('int')
    
    # Generate target pattern once (shared across all simulations)
    print("Generating targets...")
    target, phase, amp, omega, romega_vec, amp0 = generate_target(romega1=1, romega2=5, amp0=amp0)
    print("✓ Targets generated\n")
    
    # Define pqif values to simulate
    pqif_values = [0, 0.25, 0.5, 0.75, 1]
    
    # Define vt/vrest configurations (defines simulation number)
    # Each simulation uses different reset parameters for LIF neurons
    # QIF neurons always use vt=None, vrest=None (handled in dynamics)
    configs = [
        {'vt': 0, 'vrest': -17},    # Simulation 1
        {'vt': 0, 'vrest': -12.3},  # Simulation 2
    ]
    
    print(f"Total simulations (vt/vrest configurations): {len(configs)}")
    print(f"pqif values per simulation: {len(pqif_values)}")
    print(f"Seeds per pqif: {cant_seed}")
    print(f"Parallelization strategy: {len(pqif_values)} pqif values × {cant_seed} seeds = {len(pqif_values)*cant_seed} parallel processes per simulation\n")
    
    # Iterate over each vt/vrest configuration
    for num_simulacion, config in enumerate(configs, start=1):
        vt = config['vt']
        vrest = config['vrest']
        
        print(f"\n{'#'*70}")
        print(f"# STARTING SIMULATION {num_simulacion}: vt={vt}, vrest={vrest}")
        print(f"{'#'*70}\n")
        
        # Create folders and files for this simulation
        nombre_carpeta, sub_act, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes = crear_carpetas(num_simulacion)
        
        # Save targets for each pqif (all share the same target)
        for pqif in pqif_values:
            save_target(target, phase=phase, omega=omega, romega_vec=romega_vec, 
                       amp=amp, amp0=amp0, num_simulacion=num_simulacion, 
                       nombre_carpeta=nombre_carpeta, pqif=pqif)
        
        # Create parameters file
        filename_resultados = f'simulation_{num_simulacion}_results.csv'
        crear_archivo_parametros(filename_resultados, num_simulacion, 
                                nombre_carpeta, b, vt, vrest=vrest)
        
        # Create results file with header
        csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pqif', 'seed', 'nloop', 'sigma2', 'tau_rec',
                           'tau_div', 'tau_con', 'tau_chn'])
        
        # Parallelize over pqif for this simulation
        # Each pqif will internally parallelize over seeds
        results = Parallel(n_jobs=len(pqif_values), verbose=10)(
            delayed(run_pqif_simulation)(
                pqif, num_simulacion, vt, vrest, target, phase, amp, omega, romega_vec, amp0,
                N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                dt, itmax, itstim, amp_corriente, alpha, sigman,
                b, iout
            )
            for pqif in pqif_values
        )
        
        print(f"\n{'#'*70}")
        print(f"# ✓ SIMULATION {num_simulacion} COMPLETED")
        print(f"{'#'*70}\n")
    
    print("\n" + "="*70)
    print("¡ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)