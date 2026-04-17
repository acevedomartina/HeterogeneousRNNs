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
cant_seed = 50
ftrain = 1


iout = np.linspace(0, N, num=N, endpoint=False).astype('int')


# ========== FUNCIONES DE ORGANIZACIÓN ==========
def crear_subcarpeta(carpeta_padre, nombre_subcarpeta):
    ruta = os.path.join(carpeta_padre, nombre_subcarpeta)
    if not os.path.exists(ruta):
        os.makedirs(ruta, exist_ok=True)
    return ruta


def crear_carpetas(num_simulacion, omegagauss=None, sg_index=None, ts=None):
    """
    Estructura:
      simulacion_N/
        omega_X_sg_Y/         
          matrices_pesos/
          corrientes/
          inputs/
          outputs/
          nspikes/
    """
    nombre_carpeta = f"simulacion_{num_simulacion}"
    os.makedirs(nombre_carpeta, exist_ok=True)

    if omegagauss is not None and sg_index is not None and ts is not None:
        # Nombre único por combinación de parámetros del sweep
        tag = f"omega_{omegagauss}_sg_{sg_index}"
        carpeta_sweep = crear_subcarpeta(nombre_carpeta, tag)

        sub_pesos      = crear_subcarpeta(carpeta_sweep, "matrices_pesos")
        sub_corrientes = crear_subcarpeta(carpeta_sweep, "corrientes")
        sub_inputs     = crear_subcarpeta(carpeta_sweep, "inputs")
        sub_outputs    = crear_subcarpeta(carpeta_sweep, "outputs")
        sub_nspikes    = crear_subcarpeta(carpeta_sweep, "nspikes")

        return nombre_carpeta, carpeta_sweep, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes
    else:
        # Fallback sin sweep (mantiene compatibilidad con llamadas antiguas)
        sub_pesos      = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_matrices_pesos")
        sub_corrientes = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_corrientes")
        sub_inputs     = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_inputs")
        sub_outputs    = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_outputs")
        sub_nspikes    = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_nspikes")

        return nombre_carpeta, None, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes


def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta,
                              vt, vrest, omegagauss, sg_index):
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
        'vrest': [vrest],
        'sg_index': [sg_index],
        'omegagauss': [omegagauss],
        'results_file': [filename_resultados],
    }

    df = pd.DataFrame(data_parametros)
    filename_parametros = f'simulacion_{num_simulacion}_parametros.csv'
    csv_parametros_path = os.path.join(nombre_carpeta, filename_parametros)
    df.to_csv(csv_parametros_path, index=False)


# ========== GENERACIÓN DE TARGETS ==========
def generate_target(num_simulacion, nombre_carpeta, sg_index, omegagauss, amp0, pqif):
    """
    Genera el target y guarda:
      - simulation_N_targets_<pqif>.csv          (matriz NxT)
      - simulation_N_targets_parameters_<pqif>.csv  (parámetros usados)

    Se guarda un archivo de parámetros **por cada pqif** para que quede
    registro claro de qué parámetros corresponden a cada fracción QIF.
    """
    target = np.zeros(shape=(N, itmax))

    gg = np.zeros(N)
    sg = sg_index * N  # ancho de la gaussiana relativo al tamaño del sistema

    neuron_permutation = np.random.permutation(N)
    for i in range(N):
        gg[i] = amp0 * np.exp(-(i - N / 2) ** 2 / (2 * sg ** 2))

    neuron_permutation = np.random.permutation(N)

    for it in range(itmax):
        gg_shifted = np.roll(gg, int(omegagauss * it))
        target[:, it] = gg_shifted[neuron_permutation]

    # --- Guardar matriz del target ---
    target_df = pd.DataFrame(target.T, columns=[f'Neuron_{i}' for i in range(N)])
    nombre_archivo_target = f'simulation_{num_simulacion}_targets_{pqif}.csv'
    csv_target_path = os.path.join(nombre_carpeta, nombre_archivo_target)
    target_df.to_csv(csv_target_path, index=False)

    # --- Guardar parámetros del target (un archivo por pqif) ---
    data = {
        'sg_index':        [sg_index],
        'omegagauss':      [omegagauss],
        'amp0':            [amp0],
        'pqif':            [pqif],
        'sequence_order':  [list(neuron_permutation)],
    }
    df = pd.DataFrame(data)
    nombre_archivo_params = f'simulation_{num_simulacion}_targets_parameters_{pqif}.csv'
    csv_params_path = os.path.join(nombre_carpeta, nombre_archivo_params)
    df.to_csv(csv_params_path, index=False)

    return target


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
    dpi = np.mod(np.pi - np.mod(x, 2 * np.pi), 2 * np.pi)
    ispike_qif = np.where((xnew[:nqif] - x[:nqif]) > 0) and np.where(
        (xnew[:nqif] - x[:nqif] - dpi[:nqif]) > 0)
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

    ww  = np.matmul(w, w)
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
def run_simulation_omega_ts(omegagauss, sg_index, ts, pqif, num_simulacion,
                            vt, vrest, amp0,
                            N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                            dt, itmax, itstim, amp_corriente, alpha, sigman, iout):
    """
    Ejecuta la simulación para una combinación específica de
    (omegagauss, sg_index, ts, pqif)
    """

    b = 1 / ts


    print(f"Sim {num_simulacion} | omega={omegagauss} sg={sg_index} ts={ts} pqif={pqif}")
    print(f"vt={vt} vrest={vrest} b={b:.4f}")


    # Carpetas específicas para (omegagauss, sg_index)
    (nombre_carpeta, carpeta_sweep,
     sub_pesos, sub_corrientes,
     sub_inputs, sub_outputs, sub_nspikes) = crear_carpetas(
        num_simulacion, omegagauss, sg_index, ts
    )

    # Target (guardado con parámetros propios de este pqif)
    target = generate_target(
        num_simulacion=num_simulacion,
        nombre_carpeta=carpeta_sweep,   # dentro de la subcarpeta del sweep
        sg_index=sg_index,
        omegagauss=omegagauss,
        amp0=amp0,
        pqif=pqif
    )

    # Archivo de corrientes
    path_corrientes_seed = os.path.join(
        sub_corrientes, f'corrientes_pqif_{pqif}.csv'
    )
    with open(path_corrientes_seed, mode='w', newline='') as file_:
        writer_ = csv.writer(file_)
        writer_.writerow(['omegagauss', 'sg_index', 'ts', 'pqif', 'seed',
                          'iloop', 'it',
                          'II_0', 'v_0', 'II_1', 'v_1',
                          'II_N2+1', 'v_N2+1', 'II_N2+2', 'v_N2+2'])

    nqif = int(N * pqif)

    for seed in range(cant_seed):
        print(f"  seed {seed + 1}/{cant_seed}")

        np.random.seed(seed=seed)

        x, r, nspike = initialize_neurons(N)
        Iext = currents(N, itmax)

        path_Iext = os.path.join(
            carpeta_sweep, f'Iext_pqif_{pqif}_seed_{seed}.csv'
        )
        guardar_matriz_csv(Iext, path_Iext)

        w = initialize_connectivity_matrix(N, p, gsyn)
        norm_w0 = np.linalg.norm(w)
        P, idx = initialize_training(N, w)

        filename_dw = os.path.join(
            carpeta_sweep, f'dw_pqif_{pqif}_seed_{seed}.csv'
        )

        with open(filename_dw, mode='w', newline='') as file_dw:
            csv_writer_dw = csv.writer(file_dw)
            csv_writer_dw.writerow(['modt', 'modw'])

            for iloop in range(nloop):

                outputs_loop    = []
                inputs_loop     = []
                nspikes_loop    = []
                corrientes_buffer = []

                path_inputs  = os.path.join(sub_inputs,
                    f'inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                path_nspikes = os.path.join(sub_nspikes,
                    f'nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                path_outputs = os.path.join(sub_outputs,
                    f'outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')

                for it in range(itmax):
                    nspike = np.zeros(N)

                    x, r, nspike, rout, II, v = evolution(
                        x, r, Iext, w, nqif, it, dt,
                        iout, nspike, b, vt=vt, vrest=vrest
                    )

                    entrada = II + v

                    outputs_loop.append(rout)
                    inputs_loop.append(entrada)
                    nspikes_loop.append(nspike)

                    if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                        corrientes_buffer.append([
                            omegagauss, sg_index, ts, pqif, seed, iloop, it,
                            II[0], v[0], II[1], v[1],
                            II[N2 + 1], v[N2 + 1], II[N2 + 2], v[N2 + 2]
                        ])

                    if iloop > 0 and iloop <= nloop_train and int(it > itstim):
                        w, P = learning(it, iloop, w, r, P, idx, target,
                                        norm_w0, csv_writer_dw)

                np.savetxt(path_inputs,  np.array(inputs_loop),  delimiter=',')
                np.savetxt(path_nspikes, np.array(nspikes_loop), delimiter=',')
                np.savetxt(path_outputs, np.array(outputs_loop), delimiter=',')

                if corrientes_buffer:
                    with open(path_corrientes_seed, 'a', newline='') as f_corr:
                        writer_corr = csv.writer(f_corr)
                        writer_corr.writerows(corrientes_buffer)

                sigma2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)

                if iloop == 0 or iloop == (nloop_train + 1):
                    path_w_seed = os.path.join(sub_pesos,
                        f'pesos_pqif_{pqif}_matriz_iloop_{iloop}_semilla_{seed}')
                    guardar_matriz_csv(w, path_w_seed)

                # Resultados en el archivo de la simulación principal
                filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
                csv_file_path = os.path.join(nombre_carpeta, filename_resultados)

                with open(csv_file_path, 'a', newline='') as file_res:
                    writer_res = csv.writer(file_res)
                    writer_res.writerow([omegagauss, sg_index, ts, b, pqif,
                                         seed, iloop,
                                         sigma2, tau_rec, tau_div, tau_con, tau_chn])

    return num_simulacion



if __name__ == '__main__':


    # ---- Parameter sweep ----
    omegagauss_values = [0.1, 0.2]
    sg_index_values   = [0.1, 0.15]

    # Valores fijos de ts y pqif
    ts_values   = [5]
    pqif_values = [0, 0.25, 0.5, 0.75, 1]

    # Configuraciones vt / vrest
    configs = [
        {'vt': 0, 'vrest': -12.3},
        {'vt': 0, 'vrest': -17},
        {'vt': 0, 'vrest': -8.5},
        {'vt': 0, 'vrest': -22},
    ]

    n_sweep = len(omegagauss_values) * len(sg_index_values)
    n_combo = n_sweep * len(ts_values) * len(pqif_values)


    for num_simulacion, config in enumerate(configs, start=1):
        vt    = config['vt']
        vrest = config['vrest']

        nombre_carpeta = f"simulacion_{num_simulacion}"
        os.makedirs(nombre_carpeta, exist_ok=True)

        # Crear un archivo de parámetros por combinación del sweep
        for omegagauss in omegagauss_values:
            for sg_index in sg_index_values:
                tag = f"omega_{omegagauss}_sg_{sg_index}"
                carpeta_sweep = os.path.join(nombre_carpeta, tag)
                os.makedirs(carpeta_sweep, exist_ok=True)

                filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
                crear_archivo_parametros(
                    filename_resultados, num_simulacion,
                    carpeta_sweep, vt, vrest, omegagauss, sg_index
                )

        # Archivo de resultados global (con header extendido)
        filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
        csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['omegagauss', 'sg_index', 'ts', 'b', 'pqif',
                             'seed', 'nloop',
                             'sigma2', 'tau_rec', 'tau_div', 'tau_con', 'tau_chn'])

        # Todas las combinaciones del sweep
        combinaciones = [
            (omega, sg, ts, pqif)
            for omega in omegagauss_values
            for sg    in sg_index_values
            for ts    in ts_values
            for pqif  in pqif_values
        ]


        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_simulation_omega_ts)(
                omega, sg, ts, pqif,
                num_simulacion, vt, vrest, amp0,
                N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                dt, itmax, itstim, amp_corriente, alpha, sigman, iout
            )
            for omega, sg, ts, pqif in combinaciones
        )

