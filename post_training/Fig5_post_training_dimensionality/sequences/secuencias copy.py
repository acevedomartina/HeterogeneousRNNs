import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os
from joblib import Parallel, delayed


# ========== PARÁMETROS GLOBALES ==========
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
cant_seed = 1
ts = 5
b = 1 / ts
ftrain = 1
sg_index = 0.1

iout = np.linspace(0, N, num=N, endpoint=False).astype('int')


# ========== FUNCIONES DE ORGANIZACIÓN ==========
def crear_subcarpeta(carpeta_padre, nombre_subcarpeta):
    ruta = os.path.join(carpeta_padre, nombre_subcarpeta)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    return ruta


def crear_carpetas(num_simulacion, omegagauss=None): 
    nombre_carpeta = f"simulacion_{num_simulacion}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    
    # Si se especifica omegagauss, crear subcarpeta específica
    if omegagauss is not None:
        carpeta_omega = crear_subcarpeta(nombre_carpeta, f"omega_{omegagauss}")
        
 
        sub_pesos = crear_subcarpeta(carpeta_omega, f"matrices_pesos")
        sub_corrientes = crear_subcarpeta(carpeta_omega, f"corrientes")
        sub_inputs = crear_subcarpeta(carpeta_omega, f"inputs")
        sub_outputs = crear_subcarpeta(carpeta_omega, f"outputs")
        sub_nspikes = crear_subcarpeta(carpeta_omega, f"nspikes")
        
        return nombre_carpeta, carpeta_omega, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes
    else:

        sub_pesos = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_matrices_pesos")
        sub_corrientes = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_corrientes")
        sub_inputs = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_inputs")
        sub_outputs = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_outputs")
        sub_nspikes = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_nspikes")

        return nombre_carpeta, None, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes


def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta, b, vt, vrest):
    data_parametros = {
        'N': [N],
        'p': [p],
        'gsyn': [gsyn],
        'nloop': [nloop],
        'nloop_train': [nloop_train],
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
        'sg_index': [sg_index],
        'results_file': [filename_resultados],
    }

    df = pd.DataFrame(data_parametros)
    filename_parametros = f'simulacion_{num_simulacion}_parametros.csv'
    csv_parametros_path = os.path.join(nombre_carpeta, filename_parametros)
    df.to_csv(csv_parametros_path, index=False)


# ========== GENERACIÓN DE TARGETS ==========
def generate_target(sg_index, amp0, omegagauss):
    """
    Genera target gaussiano móvil (secuencia periódica)
    """
    target = np.zeros(shape=(N, itmax))
    
    gg = np.zeros(N)
    sg = sg_index * N  # ancho de la gaussiana relativo al tamaño del sistema

    
    for i in range(N):
        gg[i] = amp0 * np.exp(-(i - N/2)**2 / (2 * sg**2))
    
    for it in range(itmax):
        target[:, it] = np.roll(gg, int(omegagauss * it))
    
    return target, sg_index, omegagauss


def save_target(target, sg_index, omegagauss, amp0, num_simulacion, carpeta_omega, pqif):
    """
    Guarda el target y sus parámetros en la carpeta omega específica
    """
    target_df = pd.DataFrame(target.T, columns=[f'Neurona_{i}' for i in range(N)])
    nombre_archivo_target = f'targets_pqif_{pqif}.csv'
    csv_target_path = os.path.join(carpeta_omega, nombre_archivo_target)
    target_df.to_csv(csv_target_path, index=False)

    data = {'sg_index': [sg_index], 'omegagauss': [omegagauss], 'amp0': [amp0]}
    df = pd.DataFrame(data)
    nombre_archivo = f'targets_parametros_pqif_{pqif}.csv'
    csv_target_path = os.path.join(carpeta_omega, nombre_archivo)
    df.to_csv(csv_target_path, index=False)


def guardar_matriz_csv(matriz, nombre_archivo):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for fila in matriz:
            fila_lista = [str(elemento) for elemento in fila.flat]
            escritor_csv.writerow(fila_lista)


# ========== FUNCIONES DE DINÁMICA ==========
def dynamics(x_var, r_var, I_var, nqif, b):
    dx = np.zeros(N)
    I_noise_lif = np.random.randn(N - nqif) * sigman 
    I_noise_qif = np.random.randn(nqif) * sigman
    # LIF
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif
    # QIF
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif] * (1 + np.cos(x_var[:nqif])) + I_noise_qif
    dr = -b * r_var
    return dx, dr


def detect(x, xnew, rnew, nspike, nqif, b, vt, vrest):
    # LIF
    ispike_lif = np.where(x[nqif:] < vt) and np.where(xnew[nqif:] > vt)
    ispike_lif = ispike_lif[0] + nqif
    if len(ispike_lif) > 0:
        rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
        xnew[ispike_lif[:]] = vrest
        nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
    # QIF 
    dpi = np.mod(np.pi - np.mod(x, 2*np.pi), 2*np.pi)
    ispike_qif = np.where((xnew[:nqif] - x[:nqif]) > 0) and np.where((xnew[:nqif] - x[:nqif] - dpi[:nqif]) > 0)
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
    np.fill_diagonal(w, 0)
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


def motifs(w, gsyn, N):
    w = w - np.mean(w)
    
    ww = np.matmul(w, w)
    wtw = np.matmul(w.T, w)
    wwt = np.matmul(w, w.T)
    
    sigma2 = np.trace(wwt) / N
    
    tau_rec = np.trace(ww)
    tau_rec /= sigma2 * N
    
    tau_div = np.sum(wwt) - np.trace(wwt)
    tau_div /= sigma2 * N * (N - 1)
    
    tau_con = np.sum(wtw) - np.trace(wtw)
    tau_con /= sigma2 * N * (N - 1)
    
    tau_chn = 2 * (np.sum(ww) - np.trace(ww))
    tau_chn /= sigma2 * N * (N - 1)
    
    return sigma2, tau_rec, tau_div, tau_con, tau_chn


# ========== FUNCIÓN PRINCIPAL DE SIMULACIÓN ==========
def run_simulation_omega(omegagauss, pqif, num_simulacion, vt, vrest, sg_index, amp0,
                        N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                        dt, itmax, itstim, amp_corriente, alpha, sigman, b, iout):
    """
    Ejecuta la simulación para un valor específico de omegagauss y pqif
    """
    
    print(f"\n{'='*60}")
    print(f"Simulación {num_simulacion} - Procesando omegagauss={omegagauss}, pqif={pqif}")
    print(f"vt={vt}, vrest={vrest}")
    print(f"{'='*60}\n")
    
    # Generar target específico para este omegagauss
    target, _, _ = generate_target(sg_index=sg_index, amp0=amp0, omegagauss=omegagauss)
    
    # Crear carpetas específicas para este omegagauss
    nombre_carpeta, carpeta_omega, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes = crear_carpetas(num_simulacion, omegagauss)
    
    # Guardar target para esta combinación
    save_target(target, sg_index=sg_index, omegagauss=omegagauss, 
               amp0=amp0, num_simulacion=num_simulacion, 
               carpeta_omega=carpeta_omega, pqif=pqif)
    
    # Preparar archivo de corrientes
    path_corrientes_seed = os.path.join(sub_corrientes, 
                                       f'corrientes_pqif_{pqif}.csv')
    
    with open(path_corrientes_seed, mode='w', newline='') as file_:
        writer_ = csv.writer(file_)
        writer_.writerow(['omegagauss', 'pqif', 'seed', 'iloop', 'it', 'II_0', 'v_0', 
                        'II_1', 'v_1', 'II_N2+1', 'v_N2+1', 'II_N2+2', 'v_N2+2'])
    
    # Calcular nqif
    nqif = int(N * pqif)
    
    # Loop sobre seeds
    for seed in range(cant_seed):
        print(f"    - Procesando seed {seed+1}/{cant_seed}")
        
        np.random.seed(seed=seed)
        
        # Inicialización
        x, r, nspike = initialize_neurons(N)
        Iext = currents(N, itmax)
        
        # Guardar corriente externa
        path_Iext = os.path.join(carpeta_omega, 
                                f'Iext_pqif_{pqif}_seed_{seed}.csv')
        guardar_matriz_csv(Iext, path_Iext)
        
        w = initialize_connectivity_matrix(N, p, gsyn)
        norm_w0 = np.linalg.norm(w)
        P, idx = initialize_training(N, w)
        
        # Archivo dw
        filename_dw = os.path.join(carpeta_omega, 
                                  f'dw_pqif_{pqif}_seed_{seed}.csv')
        
        with open(filename_dw, mode='w', newline='') as file_dw:
            csv_writer_dw = csv.writer(file_dw)
            csv_writer_dw.writerow(['modt', 'modw'])
            
            # Loop de entrenamiento
            for iloop in range(nloop):
                
                # Pre-alocar arrays para este loop
                outputs_loop = []
                inputs_loop = []
                nspikes_loop = []
                corrientes_buffer = []
                
                # Paths para guardar
                path_inputs = os.path.join(sub_inputs, 
                                          f'inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                path_nspikes = os.path.join(sub_nspikes, 
                                           f'nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                path_outputs = os.path.join(sub_outputs, 
                                           f'outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                
                for it in range(itmax):
                    nspike = np.zeros(N)
                    
                    x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt, 
                                                         iout, nspike, b, vt=vt, vrest=vrest)
                    
                    entrada = II + v
                    
                    # Acumular en listas
                    outputs_loop.append(rout)
                    inputs_loop.append(entrada)
                    nspikes_loop.append(nspike)
                    
                    # Guardar corrientes en loops específicos
                    if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                        corrientes_buffer.append([omegagauss, pqif, seed, iloop, it, 
                                                II[0], v[0], II[1], v[1], 
                                                II[N2+1], v[N2+1], II[N2+2], v[N2+2]])
                    
                    # Aprendizaje
                    if iloop > 0 and iloop <= nloop_train and int(it > itstim):
                        w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer_dw)
                
                # Guardar datos del loop
                np.savetxt(path_inputs, np.array(inputs_loop), delimiter=',')
                np.savetxt(path_nspikes, np.array(nspikes_loop), delimiter=',')
                np.savetxt(path_outputs, np.array(outputs_loop), delimiter=',')
                
                # Guardar corrientes si hay
                if corrientes_buffer:
                    with open(path_corrientes_seed, 'a', newline='') as f_corr:
                        writer_corr = csv.writer(f_corr)
                        writer_corr.writerows(corrientes_buffer)
                
                # Calcular motifs
                sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)
                
                # Guardar matriz de pesos en loops específicos
                if iloop == 0 or iloop == (nloop_train + 1):
                    path_w_seed = os.path.join(sub_pesos, 
                                              f'pesos_pqif_{pqif}_matriz_iloop_{iloop}_semilla_{seed}')
                    guardar_matriz_csv(w, path_w_seed)
                
                # Guardar resultados (append mode - thread-safe con diferentes archivos)
                filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
                csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
                
                with open(csv_file_path, 'a', newline='') as file_res:
                    writer_res = csv.writer(file_res)
                    writer_res.writerow([omegagauss, pqif, seed, iloop, sigma2, tau_rec, 
                                       tau_div, tau_con, tau_chn])
    
    print(f"✓ omegagauss={omegagauss}, pqif={pqif} completado para simulación {num_simulacion}\n")
    
    return num_simulacion


# ========== EJECUCIÓN PRINCIPAL ==========
if __name__ == '__main__':
    
    print("="*70)
    print("INICIANDO SIMULACIONES PARALELIZADAS - MÚLTIPLES OMEGAGAUSS")
    print("="*70)
    
    # Lista de valores omegagauss a simular
    omegagauss_values = [0.2, 0.5, 1]  # Define los valores que quieras probar
    
    # Lista de valores pqif a simular
    pqif_values = [0, 0.25, 0.5, 0.75, 1]
    
    # Determinar configuraciones vt/vrest
    configs = [
        {'vt': 0, 'vrest': -12.3},   # Simulación 1
        {'vt': 0, 'vrest': -17},    # Simulación 2
    ]
    
    print(f"\nTotal de simulaciones (configuraciones vt/vrest): {len(configs)}")
    print(f"Valores de omegagauss por simulación: {len(omegagauss_values)}")
    print(f"Valores de pqif por simulación: {len(pqif_values)}")
    print(f"Total de combinaciones por simulación: {len(omegagauss_values) * len(pqif_values)}")
    print(f"Paralelizando sobre todas las combinaciones omegagauss x pqif\n")
    
    # Iterar sobre cada configuración vt/vrest
    for num_simulacion, config in enumerate(configs, start=1):
        vt = config['vt']
        vrest = config['vrest']
        
        print(f"\n{'#'*70}")
        print(f"# INICIANDO SIMULACIÓN {num_simulacion}: vt={vt}, vrest={vrest}")
        print(f"{'#'*70}\n")
        
        # Crear carpeta principal de simulación
        nombre_carpeta = f"simulacion_{num_simulacion}"
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)
        
        # Crear archivo de parámetros
        filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
        crear_archivo_parametros(filename_resultados, num_simulacion, 
                                nombre_carpeta, b, vt, vrest=vrest)
        
        # Crear archivo de resultados con header (incluyendo omegagauss)
        csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['omegagauss', 'pqif', 'seed', 'nloop', 'sigma2', 'tau_rec',
                           'tau_div', 'tau_con', 'tau_chn'])
        
        # Crear todas las combinaciones de (omegagauss, pqif)
        combinaciones = [(omega, pqif) for omega in omegagauss_values for pqif in pqif_values]
        
        # Paralelizar sobre todas las combinaciones
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_simulation_omega)(
                omega, pqif, num_simulacion, vt, vrest, sg_index, amp0,
                N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                dt, itmax, itstim, amp_corriente, alpha, sigman,
                b, iout
            )
            for omega, pqif in combinaciones
        )
        
        print(f"\n{'#'*70}")
        print(f"# ✓ SIMULACIÓN {num_simulacion} COMPLETADA")
        print(f"{'#'*70}\n")
    
    print("\n" + "="*70)
    print("¡TODAS LAS SIMULACIONES COMPLETADAS CON ÉXITO!")
    print("="*70)