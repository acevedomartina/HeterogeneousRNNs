import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os
from joblib import Parallel, delayed

# ========== GLOBAL PARAMETERS ==========
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
ts = 5
b = 1 / ts
ftrain = 1

iout = np.linspace(0, N, num=N, endpoint=False).astype('int')


# ========== FILE ORGANIZATION ==========
def create_subfolder(parent, name):
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    return path


def create_folders(num_simulation, romega1=None, romega2=None):
    """
    Structure:
      simulation_{n}/
        omega1_{r1}_omega2_{r2}/
          connectivity_matrix/
          currents/
          inputs/
          outputs/
          nspikes/
    """
    sim_folder = f"simulation_{num_simulation}"
    os.makedirs(sim_folder, exist_ok=True)

    if romega1 is not None and romega2 is not None:
        tag          = f"omega1_{romega1}_omega2_{romega2}"
        sweep_folder = create_subfolder(sim_folder, tag)

        sub_weights  = create_subfolder(sweep_folder, "connectivity_matrix")
        sub_currents = create_subfolder(sweep_folder, "currents")
        sub_inputs   = create_subfolder(sweep_folder, "inputs")
        sub_outputs  = create_subfolder(sweep_folder, "outputs")
        sub_nspikes  = create_subfolder(sweep_folder, "nspikes")

        return sim_folder, sweep_folder, sub_weights, sub_currents, sub_inputs, sub_outputs, sub_nspikes
    else:
        sub_weights  = create_subfolder(sim_folder, f"simulation_{num_simulation}_connectivity_matrix")
        sub_currents = create_subfolder(sim_folder, f"simulation_{num_simulation}_currents")
        sub_inputs   = create_subfolder(sim_folder, f"simulation_{num_simulation}_inputs")
        sub_outputs  = create_subfolder(sim_folder, f"simulation_{num_simulation}_outputs")
        sub_nspikes  = create_subfolder(sim_folder, f"simulation_{num_simulation}_nspikes")

        return sim_folder, None, sub_weights, sub_currents, sub_inputs, sub_outputs, sub_nspikes


def create_parameters_file(filename_results, num_simulation, sim_folder,
                            b, vt, vrest, romega1, romega2):
    data = {
        'N':             [N],
        'p':             [p],
        'gsyn':          [gsyn],
        'nloop':         [nloop],
        'nloop_train':   [nloop_train],
        'cant_seed':     [cant_seed],
        'dt':            [dt],
        'itmax':         [itmax],
        'itstim':        [itstim],
        'amp_corriente': [amp_corriente],
        'amp0':          [amp0],
        'ftrain':        [ftrain],
        'alpha':         [alpha],
        'sigman':        [sigman],
        'vt':            [vt],
        'b':             [b],
        'vrest':         [vrest],
        'romega1':       [romega1],
        'romega2':       [romega2],
        'results_file':  [filename_results],
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(sim_folder,
              f'simulation_{num_simulation}_parameters.csv'), index=False)


def save_matrix_csv(matrix, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([str(x) for x in row.flat])


# ========== TARGET GENERATION ==========
def generate_target(romega1, romega2, amp0):
    target = np.zeros((N, itmax))
    amp    = np.random.uniform(size=N) * amp0
    phase  = np.random.uniform(0, 2*np.pi, size=N)

    indices    = np.random.permutation(N)
    romega_vec = np.zeros(N)
    for i in range(N2):
        romega_vec[indices[i]]      = romega1
        romega_vec[indices[i + N2]] = romega2

    omega = romega_vec * 2 * np.pi / itmax

    for it in range(itmax):
        target[:, it] = amp * np.cos(it * omega + phase)

    return target, amp, phase, omega, romega_vec, amp0


def save_target(target, phase, omega, romega_vec, amp, amp0,
                num_simulation, folder, pqif):
    # Parameters file
    data = {
        'Neuron':  range(N),
        'Phase':   phase,
        'Frequency': omega,
        'romega':  romega_vec,
        'Amplitude': amp,
        'amp0':    amp0,
    }
    pd.DataFrame(data).to_csv(
        os.path.join(folder, f'simulation_{num_simulation}_targets_parameters.csv'),
        index=False)

    # Target matrix
    pd.DataFrame(target.T, columns=[f'Neuron_{i}' for i in range(N)]).to_csv(
        os.path.join(folder, f'simulation_{num_simulation}_targets_{pqif}.csv'),
        index=False)


# ========== DYNAMICS ==========
def dynamics(x_var, r_var, I_var, nqif, b):
    dx = np.zeros(N)
    I_noise_lif = np.random.randn(N - nqif) * sigman
    I_noise_qif = np.random.randn(nqif) * sigman
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif
    dx[:nqif] = (1 - np.cos(x_var[:nqif])
                 + I_var[:nqif] * (1 + np.cos(x_var[:nqif]))
                 + I_noise_qif)
    dr = -b * r_var
    return dx, dr


def detect(x, xnew, rnew, nspike, nqif, b, vt, vrest):
    # LIF
    ispike_lif = np.where((x[nqif:] < vt) & (xnew[nqif:] > vt))[0] + nqif
    if len(ispike_lif) > 0:
        rnew[ispike_lif]   += b
        xnew[ispike_lif]    = vrest
        nspike[ispike_lif] += 1
    # QIF
    dpi = np.mod(np.pi - np.mod(x, 2*np.pi), 2*np.pi)
    ispike_qif = np.where(
        ((xnew[:nqif] - x[:nqif]) > 0) &
        ((xnew[:nqif] - x[:nqif] - dpi[:nqif]) > 0)
    )[0]
    if len(ispike_qif) > 0:
        rnew[ispike_qif]   += b
        nspike[ispike_qif] += 1
    return xnew, rnew, nspike


def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vrest):
    II = np.squeeze(np.asarray(Iext[:, it]))
    v  = w.dot(r.T).A1
    dx, dr   = dynamics(x, r, II + v, nqif, b)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr   = dynamics(xnew, rnew, II + v, nqif, b)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vrest)
    return np.copy(xnew), np.copy(rnew), nspike, rnew[iout], II, v


def initialize_connectivity_matrix(N, p, gsyn):
    w = sparse.random(N, N, p, data_rvs=np.random.randn).todense()
    np.fill_diagonal(w, 0)
    w *= gsyn / np.sqrt(p * N)
    for i in range(N):
        i0 = np.where(w[i, :])[1]
        if len(i0) > 0:
            w[i, i0] -= np.sum(w[i, i0]) / len(i0)
    return w


def initialize_neurons(N):
    return np.random.uniform(size=N) * 2*np.pi, np.zeros(N), np.zeros(N)


def initialize_training(N, w):
    idx, P = [], []
    for i in range(N):
        ind = np.where(w[i, :])[1]
        idx.append(ind)
        P.append(np.identity(len(ind)) / alpha)
    return P, idx


def currents(N, itmax):
    Iext = np.zeros((N, itmax))
    Ibac = amp_corriente * (2*np.random.uniform(size=N) - 1)
    Iext[:, :itstim] = Ibac[:, None]
    return Iext


def learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer):
    error = target[:, it:it+1] - w @ r.reshape(N, 1)
    for i in range(N):
        ri  = r[idx[i]].reshape(len(idx[i]), 1)
        k1  = P[i] @ ri
        k2  = ri.T @ P[i]
        P[i] -= (k1 @ k2) / (1 + ri.T @ k1)
        w[i, idx[i]] += error[i, 0] * P[i] @ r[idx[i]]
    if it % 10 == 0:
        csv_writer.writerow([it + iloop*itmax,
                             np.log(np.linalg.norm(w) / norm_w0)])
    return w, P


def motifs(w, gsyn, N):
    w   = w - np.mean(w)
    ww  = np.matmul(w, w)
    wtw = np.matmul(w.T, w)
    wwt = np.matmul(w, w.T)
    s2  = np.trace(wwt) / N
    tau_rec = np.trace(ww)          / (s2 * N)
    tau_div = (np.sum(wwt) - np.trace(wwt)) / (s2 * N * (N-1))
    tau_con = (np.sum(wtw) - np.trace(wtw)) / (s2 * N * (N-1))
    tau_chn = 2*(np.sum(ww) - np.trace(ww)) / (s2 * N * (N-1))
    return s2, tau_rec, tau_div, tau_con, tau_chn


# ========== SINGLE SEED WORKER ==========
def run_single_seed(seed, pqif, num_simulation, vt, vrest, target,
                    N, N2, p, gsyn, nloop, nloop_train,
                    dt, itmax, itstim, amp_corriente, alpha, sigman,
                    b, iout, sim_folder, sweep_folder,
                    sub_weights, sub_currents, sub_inputs, sub_outputs, sub_nspikes):

    nqif = int(N * pqif)
    np.random.seed(seed=seed)

    x, r, nspike = initialize_neurons(N)
    Iext = currents(N, itmax)

    save_matrix_csv(Iext, os.path.join(sweep_folder,
        f'simulation_{num_simulation}_Iext_pqif_{pqif}_seed_{seed}.csv'))

    w       = initialize_connectivity_matrix(N, p, gsyn)
    norm_w0 = np.linalg.norm(w)
    P, idx  = initialize_training(N, w)

    filename_dw = os.path.join(sweep_folder,
        f'simulation_{num_simulation}_dw_pqif_{pqif}_seed_{seed}.csv')

    seed_results      = []
    currents_buffer   = []

    with open(filename_dw, 'w', newline='') as file_dw:
        dw_writer = csv.writer(file_dw)
        dw_writer.writerow(['modt', 'modw'])

        for iloop in range(nloop):
            outputs_loop, inputs_loop, nspikes_loop = [], [], []

            path_inputs  = os.path.join(sub_inputs,
                f'inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
            path_outputs = os.path.join(sub_outputs,
                f'outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
            path_nspikes = os.path.join(sub_nspikes,
                f'nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')

            for it in range(itmax):
                nspike = np.zeros(N)
                x, r, nspike, rout, II, v = evolution(
                    x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vrest)

                outputs_loop.append(rout)
                inputs_loop.append(II + v)
                nspikes_loop.append(nspike)

                if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                    currents_buffer.append([pqif, seed, iloop, it,
                                            II[0], v[0], II[1], v[1],
                                            II[N2+1], v[N2+1], II[N2+2], v[N2+2]])

                if 0 < iloop <= nloop_train and it > itstim:
                    w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, dw_writer)

            np.savetxt(path_inputs,  np.array(inputs_loop),  delimiter=',')
            np.savetxt(path_outputs, np.array(outputs_loop), delimiter=',')
            np.savetxt(path_nspikes, np.array(nspikes_loop), delimiter=',')

            s2, tau_rec, tau_div, tau_con, tau_chn = motifs(w, gsyn, N)

            if iloop == 0 or iloop == nloop_train + 1:
                save_matrix_csv(w, os.path.join(sub_weights,
                    f'simulation_{num_simulation}_connectivity_pqif_{pqif}'
                    f'_iloop_{iloop}_seed_{seed}'))

            seed_results.append([pqif, seed, iloop,
                                  s2, tau_rec, tau_div, tau_con, tau_chn])

    return seed_results, currents_buffer


# ========== PER-PQIF WORKER (parallelizes over seeds) ==========
def run_pqif_simulation(pqif, romega1, romega2, num_simulation, vt, vrest,
                        target, amp0,
                        N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                        dt, itmax, itstim, amp_corriente, alpha, sigman,
                        b, iout):

    print(f"\nSim {num_simulation} | ω1={romega1} ω2={romega2} "
          f"pqif={pqif} | vt={vt} vrest={vrest}")

    sim_folder    = f"simulation_{num_simulation}"
    tag           = f"omega1_{romega1}_omega2_{romega2}"
    sweep_folder  = os.path.join(sim_folder, tag)
    sub_weights   = os.path.join(sweep_folder, "connectivity_matrix")
    sub_currents  = os.path.join(sweep_folder, "currents")
    sub_inputs    = os.path.join(sweep_folder, "inputs")
    sub_outputs   = os.path.join(sweep_folder, "outputs")
    sub_nspikes   = os.path.join(sweep_folder, "nspikes")

    # Currents file header
    path_currents_pqif = os.path.join(sub_currents,
        f'simulation_{num_simulation}_currents_pqif_{pqif}.csv')
    with open(path_currents_pqif, 'w', newline='') as f:
        csv.writer(f).writerow(['pqif', 'seed', 'iloop', 'it',
                                 'II_0', 'v_0', 'II_1', 'v_1',
                                 'II_N2+1', 'v_N2+1', 'II_N2+2', 'v_N2+2'])

    # Parallelize over seeds
    results = Parallel(n_jobs=cant_seed, verbose=5)(
        delayed(run_single_seed)(
            seed, pqif, num_simulation, vt, vrest, target,
            N, N2, p, gsyn, nloop, nloop_train,
            dt, itmax, itstim, amp_corriente, alpha, sigman,
            b, iout, sim_folder, sweep_folder,
            sub_weights, sub_currents, sub_inputs, sub_outputs, sub_nspikes
        )
        for seed in range(cant_seed)
    )

    all_results, all_currents = [], []
    for seed_results, currents_buffer in results:
        all_results.extend(seed_results)
        all_currents.extend(currents_buffer)

    # Write results
    csv_file = os.path.join(sim_folder,
                            f'simulation_{num_simulation}_results.csv')
    with open(csv_file, 'a', newline='') as f:
        csv.writer(f).writerows(all_results)

    if all_currents:
        with open(path_currents_pqif, 'a', newline='') as f:
            csv.writer(f).writerows(all_currents)

    return num_simulation


if __name__ == '__main__':

    romega1_values = [1, 5]
    romega2_values = [1, 5]
    pqif_values    = [0, 0.25, 0.5, 0.75, 1]

    configs = [
        {'vt': 0, 'vrest': -12.3},
        {'vt': 0, 'vrest': -17},
        {'vt': 0, 'vrest': -8.5},
        {'vt': 0, 'vrest': -22},
    ]

    for num_simulation, config in enumerate(configs, start=1):
        vt    = config['vt']
        vrest = config['vrest']

        sim_folder = f"simulation_{num_simulation}"
        os.makedirs(sim_folder, exist_ok=True)

        csv_file = os.path.join(sim_folder,
                                f'simulation_{num_simulation}_results.csv')
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['pqif', 'seed', 'nloop',
                                    'sigma2', 'tau_rec', 'tau_div',
                                    'tau_con', 'tau_chn'])

        combinations = [
            (r1, r2, pqif)
            for r1   in romega1_values
            for r2   in romega2_values
            for pqif in pqif_values
        ]

        # ── pre-generar y guardar un target fijo por (r1, r2) ──────────────
        # Se genera UNA SOLA VEZ con seed fija y se reutiliza para todos
        # los pqif de ese sweep point, tanto en disco como en el worker.
        target_cache = {}
        for r1, r2 in [(r1, r2) for r1 in romega1_values for r2 in romega2_values]:

            # crear carpetas
            (sim_folder_ret, sweep_folder,
             sub_weights, sub_currents,
             sub_inputs, sub_outputs, sub_nspikes) = create_folders(
                num_simulation, r1, r2)

            # seed fija y única para este (r1, r2)
            np.random.seed(0)
            target, amp, phase, omega, romega_vec, amp0_used = generate_target(
                r1, r2, amp0)

            # guardar para todos los pqif
            for pqif in pqif_values:
                save_target(target, phase, omega, romega_vec, amp, amp0_used,
                            num_simulation, sweep_folder, pqif)

            create_parameters_file(
                f'simulation_{num_simulation}_results.csv',
                num_simulation, sweep_folder,
                b, vt, vrest, r1, r2)

            # cachear el array para pasarlo al worker
            target_cache[(r1, r2)] = target

        # ── Parallel: pasar el target cacheado, no regenerarlo ─────────────
        Parallel(n_jobs=-1, verbose=10)(
            delayed(run_pqif_simulation)(
                pqif, r1, r2, num_simulation, vt, vrest,
                target_cache[(r1, r2)],   # <-- mismo target que se guardó
                amp0,
                N, N2, p, gsyn, nloop, nloop_train, cant_seed,
                dt, itmax, itstim, amp_corriente, alpha, sigman,
                b, iout
            )
            for r1, r2, pqif in combinations
        )
