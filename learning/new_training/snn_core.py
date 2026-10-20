import numpy as np 
from scipy import sparse

# NOW: (CAMBIO) Noise is generated in evolution() once per time step, not inside dynamics().
def dynamics(x_var,r_var,I_var,nqif, b, N):
    '''
    Dynamics of neurons.
    Compute derivatives of neuron state and adaptation variable.
    LIF neurons: neurons from index nqif onward
    QIF neurons: neurons from index 0 to nqif-1

    LIF dynamics: dx/dt = -x + I
    QIF dynamics: dx/dt = 1 - cos(x) + I*(1 + cos(x))
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

    # Compute derivative for LIF neurons
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:]

    # Compute derivative for QIF neurons
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif]))
    
    # Compute derivative for adaptation variable r
    dr = -b*r_var 

    return dx,dr

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

def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vreset, rng, N, sigman):
    II = np.squeeze(np.asarray(Iext[:, it]))
    v = w.dot(r.T).A1
    # NOW: (CAMBIO) Generate one noise realization per time step and reuse it in both RK2 stages.
    Itot = II + v + sigman * rng.standard_normal(N)
    dx, dr = dynamics(x, r, Itot, nqif, b, N)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr = dynamics(xnew, rnew, Itot, nqif, b, N)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vreset)
    x, r = np.copy(xnew), np.copy(rnew)

    return x, r, nspike, r[iout], II, v

def initialize_neurons(N, nqif, rng, vreset, vt):
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
    # NOW: (CAMBIO) Initialize QIF phases and LIF voltages in their respective state spaces.
    x = np.zeros(N)
    if nqif > 0:
        x[:nqif] = rng.uniform(0, 2 * np.pi, size=nqif)
    if nqif < N:
        x[nqif:] = rng.uniform(vreset, vt, size=N - nqif)
    r = np.zeros(N)
    nspike = np.zeros(N)
    return x, r, nspike

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

def initialize_training(N, w, alpha):
    """Initialize RLS correlation matrices and connectivity indices."""
    idx = []
    P = []
    for i in range(N):
        ind = np.flatnonzero(np.asarray(w[i, :]).ravel()).astype(np.int64)
        idx.append(ind)
        P.append(np.eye(len(ind), dtype=np.float64) / alpha)
    return P, idx

def currents(N, itmax, rng, current_amplitude, IEXT_LOW_FRAC, IEXT_HIGH_FRAC, itstim):
    Iext=np.zeros((N,itmax))
    # NOW: (CAMBIO) Sample initial currents from a configurable signed/asymmetric interval.
    Ibac = current_amplitude * (
        IEXT_LOW_FRAC
        + (IEXT_HIGH_FRAC - IEXT_LOW_FRAC) * rng.uniform(size=N)
    )
    Iext[:, :itstim] = Ibac[:, None]  # Vectorized assignment
    return Iext