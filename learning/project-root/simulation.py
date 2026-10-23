
"""
Network dynamics, learning, and single-seed simulations.

Responsibilities:
    - Initialize neurons and connectivity
    - Generate external currents
    - Evolve LIF/QIF network dynamics
    - Apply recursive least squares (RLS) learning
    - Run one network realization across training loops
    - Collect and save selected simulation results

Experiment sweeps and parallelization belong in experiment.py.
Configuration belongs in config.py.
File-writing conventions belong in storage.py.
"""

import numpy as np
from scipy import sparse
from numba import njit

from storage import (
    save_activity,
    save_connectivity,
    save_dw,
)



# Initialization


# def initialize_neurons(N, rng):
#     """
#     Initialize neuron states.

#     Returns
#     ---
#     x : ndarray
#         Initial membrane state / phase.

#     r : ndarray
#         Initial neuronal output.

#     nspike : ndarray
#         Spike counts.
#     """

#     x = rng.uniform(size=N) * 2 * np.pi
#     r = np.zeros(N)
#     nspike = np.zeros(N)

#     return x, r, nspike

def initialize_neurons(cfg, nqif, rng, vt, vreset):
    """Initialize QIF phases and LIF voltages in their respective state spaces."""
    x = np.zeros(cfg.N)
    if nqif > 0:
        x[:nqif] = rng.uniform(0, 2 * np.pi, size=nqif)
    if nqif < cfg.N:
        x[nqif:] = rng.uniform(vreset, vt, size=cfg.N - nqif)
    r = np.zeros(cfg.N)
    nspike = np.zeros(cfg.N)
    return x, r, nspike


def initialize_connectivity_matrix(N, p, gsyn, rng):
    """
    Initialize random recurrent connectivity.

    Preserves the original connectivity initialization:
        - Sparse random connections
        - Gaussian weights
        - No self-connections
        - Weight scaling by gsyn / sqrt(p*N)
        - Zero mean over nonzero connections in each row
    """

    w = sparse.random(
        N,
        N,
        density=p,
        random_state=rng,
        data_rvs=lambda n: rng.standard_normal(n),
    ).todense()

    np.fill_diagonal(w, 0)

    w *= gsyn / np.sqrt(p * N)

    for i in range(N):

        indices = np.where(w[i, :])[1]

        if len(indices) > 0:

            mean_weight = np.sum(w[i, indices]) / len(indices)

            w[i, indices] -= mean_weight

    return w


def initialize_training(N, w, alpha):
    """
    Initialize RLS correlation matrices and connectivity indices.
    """

    idx = []
    P = []

    for i in range(N):

        indices = np.flatnonzero(
            np.asarray(w[i, :]).ravel()
        ).astype(np.int64)

        idx.append(indices)

        P.append(
            np.eye(len(indices), dtype=np.float64) / alpha
        )

    return P, idx


# def generate_external_current(cfg, rng):
#     """
#     Generate external stimulation current.

#     The current is applied during the first cfg.itstim
#     timesteps, matching the original implementation.

#     This function does not decide whether the current
#     is shared between experiments, conditions, or seeds.
#     """

#     Iext = np.zeros((cfg.N, cfg.itmax))

#     Ibac = (cfg.current_amplitude* (2 * rng.uniform(size=cfg.N) - 1))

#     Iext[:, :cfg.itstim] = Ibac[:, None]

#     return Iext

def generate_external_current(cfg, rng):
    """Generate external currents from a configurable uniform interval."""
    Iext = np.zeros((cfg.N, cfg.itmax))
    Ibac = cfg.current_amplitude * (cfg.IEXT_LOW_FRAC + (cfg.IEXT_HIGH_FRAC - cfg.IEXT_LOW_FRAC) * rng.uniform(size=cfg.N))
    Iext[:, :cfg.itstim] = Ibac[:, None]
    return Iext

def currents(cfg, rng):
    """Updated currents() function from the update from M"""
    Iext=np.zeros((cfg.N, cfg.itmax))
    # NOW: (CAMBIO) Sample initial currents from a configurable signed/asymmetric interval.
    Ibac = cfg.current_amplitude * (cfg.IEXT_LOW_FRAC+ (cfg.IEXT_HIGH_FRAC - cfg.IEXT_LOW_FRAC) * rng.uniform(size=cfg.N))
    Iext[:, :cfg.itstim] = Ibac[:, None]  # Vectorized assignment
    return Iext




# Neuronal dynamics

def dynamics(x_var, r_var, I_var, nqif, cfg):
    """Compute deterministic LIF/QIF derivatives using the supplied total current."""
    dx = np.zeros(cfg.N)
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:]
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif] * (1 + np.cos(x_var[:nqif]))
    dr = -cfg.b * r_var
    return dx, dr


def detect(x, xnew, rnew, nspike, nqif, cfg, vt, vreset):
    """
    Detect spikes and update neuronal output.

    LIF neurons:
        Detect threshold crossing and reset state.

    QIF neurons:
        Detect crossing of phase pi.
    """

    b = cfg.b

    # LIF spike detection
    ispike_lif = np.where((x[nqif:] < vt) & (xnew[nqif:] > vt))[0]
    ispike_lif = ispike_lif + nqif

    if len(ispike_lif) > 0:
        rnew[ispike_lif] += b
        xnew[ispike_lif] = vreset
        nspike[ispike_lif] += 1

    # QIF spike detection
    dpi = np.mod(np.pi - np.mod(x[:nqif], 2 * np.pi), 2 * np.pi,)
    delta_x = xnew[:nqif] - x[:nqif]
    ispike_qif = np.where((delta_x > 0) & ((delta_x - dpi) > 0))[0]
    if len(ispike_qif) > 0:
        rnew[ispike_qif] += b
        nspike[ispike_qif] += 1

    return xnew, rnew, nspike

def evolution(x, r, Iext, w, nqif, it, cfg, vt, vreset, rng):
    """Advance the network one timestep using RK2 with shared noise across stages."""
    dt = cfg.dt

    II = np.asarray(Iext[:, it]).ravel()
    v = np.asarray(w @ r).ravel()

    # One noise realization per timestep, shared by both RK2 stages
    Itot = II + v + cfg.sigman * rng.standard_normal(cfg.N)

    # First RK2 stage
    dx, dr = dynamics(x, r, Itot, nqif, cfg)
    xmid = x + dt * dx / 2
    rmid = r + dt * dr / 2

    # Second RK2 stage, using the same total current
    dx, dr = dynamics(xmid, rmid, Itot, nqif, cfg)
    xnew = x + dt * dx
    rnew = r + dt * dr

    # Spike detection
    nspike = np.zeros(cfg.N)
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, cfg, vt, vreset)

    return xnew.copy(), rnew.copy(), nspike, II, v



# RLS learning

@njit(cache=True)
def rls_update_numba(P_i, ri, error_i):
    """
    JIT-compiled RLS update for one neuron.
    """

    n = ri.shape[0]

    k = np.empty(n, dtype=np.float64)

    for j in range(n):

        value = 0.0

        for m in range(n):
            value += P_i[j, m] * ri[m]

        k[j] = value

    den = 1.0

    for j in range(n):
        den += ri[j] * k[j]

    inv_den = 1.0 / den

    for j in range(n):
        for m in range(n):
            P_i[j, m] -= k[j] * k[m] * inv_den

    dw = np.empty(n, dtype=np.float64)

    scale = error_i * inv_den

    for j in range(n):
        dw[j] = scale * k[j]

    return dw


def learning(
    it,
    iloop,
    w,
    r,
    P,
    idx,
    target,
    norm_w0,
    dw_buffer,
    cfg,
):
    """
    Apply one RLS learning update.
    """

    N = cfg.N

    prediction = np.asarray(
        w @ r.reshape(N, 1)
    ).ravel()

    error = target[:, it] - prediction

    if cfg.save_dw:
        w1 = np.zeros(w.shape)

    for i in range(N):

        indices = idx[i]
        ri = r[indices]

        dw = rls_update_numba(
            P[i], ri, error[i]
        )

        w[i, indices] += dw

        if cfg.save_dw:
            w1[i, indices] = dw

    # Track weight-change magnitude
    if (
        cfg.save_dw
        and dw_buffer is not None
        and it % 10 == 0
    ):

        modt_value = (
            (it - cfg.itstim)
            + (iloop - 1)
            * (cfg.itmax - cfg.itstim)
        )

        modw_value = np.log(
            np.linalg.norm(w1) / norm_w0
        )

        dw_buffer.append(
            (modt_value, modw_value)
        )

    return w, P



# Network motifs (optional analysis)


def motifs(w):
    """
    Calculate network connectivity motifs.

    Returns:
        sigma2, tau_rec, tau_div, tau_con, tau_chn
    """

    w = np.asarray(w)
    N = w.shape[0]

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



# Run one network realization


def run_single_seed(
    seed,
    pqif,
    num_simulation,
    simulation_config,
    target,
    Iext,
    cfg,
    folders,
):
    """
    Run one complete network realization.

    Parameters
    --
    seed : int
        Replicate index.

    pqif : float
        Fraction of QIF neurons.

    num_simulation : int
        Network configuration index.

    simulation_config : dict
        Contains 'vt' and 'vreset'.

    target : ndarray
        Target activity, shape (N, itmax).

    Iext : ndarray
        External stimulation current.

    cfg : Config
        Experiment configuration.

    folders : dict
        Output directories created by storage.py.

    Returns
    ---
    currents_buffer : list
        Sampled currents for this seed.

    Notes
    -
    The experiment runner is responsible for collecting
    currents from all seeds and saving them as one CSV
    per pqif condition.
    """

    N = cfg.N
    N2 = cfg.N2

    vt = simulation_config["vt"]
    vreset = simulation_config["vreset"]

    
    # Random generator and network initialization
    
    rng = np.random.default_rng(
        cfg.network_seed(seed)
    )

    # nqif = int(N * pqif)

    # x, r, nspike = initialize_neurons(N, rng)

    nqif = int(N * pqif)

    x, r, nspike = initialize_neurons(
        cfg=cfg,
        nqif=nqif,
        rng=rng,
        vt=vt,
        vreset=vreset,
    )

    w = initialize_connectivity_matrix(
        N=N,
        p=cfg.p,
        gsyn=cfg.gsyn,
        rng=rng,
    )

    norm_w0 = np.linalg.norm(w)

    P, idx = initialize_training(
        N=N,
        w=w,
        alpha=cfg.alpha,
    )

    
    # Result buffers
    

    dw_buffer = [] if cfg.save_dw else None

    currents_buffer = []

    
    # Training loops
    

    for iloop in range(cfg.nloop):

        # Saving logic
        save_this_loop = (iloop in cfg.save_activity_loops)
        outputs_loop = ([] if save_this_loop and cfg.save_outputs else None)
        inputs_loop = ([] if save_this_loop and cfg.save_inputs else None)
        nspikes_loop = ([] if save_this_loop and cfg.save_nspikes else None)

        
        # Time evolution
        for it in range(cfg.itmax):

            x, r, nspike, II, v = evolution(
                x=x,
                r=r,
                Iext=Iext,
                w=w,
                nqif=nqif,
                it=it,
                cfg=cfg,
                vt=vt,
                vreset=vreset,
                rng=rng,
            )

            total_input = II + v

            
            # Collect neuronal activity
            if save_this_loop:

                if cfg.save_outputs:
                    outputs_loop.append(r.copy())

                if cfg.save_inputs:
                    inputs_loop.append(total_input.copy())

                if cfg.save_nspikes:
                    nspikes_loop.append(nspike.copy())

            
            # Sample currents
            
            if (
                iloop in [
                    cfg.nloop_train + 1,
                    cfg.nloop - 1,
                ]
                and it % 20 == 0
            ):

                currents_buffer.append([
                    pqif,
                    seed,
                    iloop,
                    it,
                    II[0], v[0],
                    II[1], v[1],
                    II[N2 + 1], v[N2 + 1],
                    II[N2 + 2], v[N2 + 2],
                ])

            
            # Learning

            if (
                0 < iloop <= cfg.nloop_train
                and it >= cfg.itstim
            ):

                w, P = learning(
                    it=it,
                    iloop=iloop,
                    w=w,
                    r=r,
                    P=P,
                    idx=idx,
                    target=target,
                    norm_w0=norm_w0,
                    dw_buffer=dw_buffer,
                    cfg=cfg,
                )

        
        # Save selected activity loops

        if save_this_loop:

            if cfg.save_outputs:

                save_activity(
                    activity=outputs_loop,
                    activity_type="outputs",
                    num_simulation=num_simulation,
                    pqif=pqif,
                    seed=seed,
                    iloop=iloop,
                    folders=folders,
                )

            if cfg.save_inputs:

                save_activity(
                    activity=inputs_loop,
                    activity_type="inputs",
                    num_simulation=num_simulation,
                    pqif=pqif,
                    seed=seed,
                    iloop=iloop,
                    folders=folders,
                )

            if cfg.save_nspikes:

                save_activity(
                    activity=nspikes_loop,
                    activity_type="nspikes",
                    num_simulation=num_simulation,
                    pqif=pqif,
                    seed=seed,
                    iloop=iloop,
                    folders=folders,
                )

        
        # Save connectivity matrices
        
        if iloop in [0, cfg.nloop_train + 1]:

            save_connectivity(
                w=w,
                num_simulation=num_simulation,
                pqif=pqif,
                seed=seed,
                iloop=iloop,
                folders=folders,
            )

    
    # Save weight-change tracking
    
    if cfg.save_dw:

        simulation_dir = folders["connectivity"].parent

        save_dw(
            dw_buffer=dw_buffer,
            num_simulation=num_simulation,
            pqif=pqif,
            seed=seed,
            simulation_dir=simulation_dir,
        )

    return currents_buffer
