import numpy as np
import matplotlib.pyplot as plt

# Script containing all the functions that are regularly used during analysis

def reorder_connectivity(M, sequence_position):
    """
    Reorder connectivity matrix according to sequence position.

    Parameters:

        M : array-like, shape (N, N)
            Connectivity matrix.
        neuron_idx : array-like, shape (N,)
            Neuron indices corresponding to the rows/columns of M.
        permutation_idx : array-like, shape (N,)
            Sequence position assigned to each neuron.
            permutation_idx[i] is the sequence position of neuron neuron_idx[i].

    Returns:

        M_ordered : ndarray, shape (N, N)
            Connectivity matrix reordered by sequence position.
        neuron_order : ndarray
            Neuron indices in sequence order.

    """

    order = np.argsort(sequence_position)
    return M[np.ix_(order, order)]


def get_profile_from_matrix(J):
    """
    Circularly shifting profile
    """

    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got {J.shape}")

    N = J.shape[0]
    J_sf = np.empty_like(J, dtype=float)

    for i in range(N):
        shift = int(np.ceil(N / 2)) - (i + 1)
        J_sf[i, :] = np.roll(J[i, :], shift)

    if N % 2 == 0:
        J_sf = np.column_stack((J_sf[:, -1], J_sf))

    mean_J = np.mean(J_sf, axis=0)
    std_J = np.std(J_sf, axis=0, ddof=1)

    neuron_distance = np.arange(-N // 2, N // 2 + 1) if N % 2 == 0 else np.arange(-(N // 2), N // 2 + 1)

    return J_sf, mean_J, std_J, neuron_distance

def load_W(connectivity_path, sim, pqif, seed):
    """Takes parent path and loads matrix from specific file"""
    W_path = connectivity_path / f"simulation_{sim}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}.npy"
    W = np.load(W_path)
    return W


def average_connectivity(connectivity_path, sim, pqif, seedrange):
    """
    Averages the connectivity matrix within one simulation*pqif, across seeds. Returns this together with its mean and std of the resulting weights.
    Uses the function load_W() to obtain matrices.

    Parameters:
        connectivity_path : string
            parent path of the files containing connectivity matrices
        sim : integer
            simulation number
        pqif : float
            fraction of QIF neurons
        seedrange : range
            range of seeds

    Returns:
        mean_W : mean connectivity matrix
        mean_W_mean_weights : mean of the weights within mean_W
        mean_W_std_weights : std of the weights within mean_W
    """

    matrices = []

    for seed in seedrange:

        W = load_W(connectivity_path, sim, pqif, seed)
        matrices.append(W)

    matrices = np.stack(matrices, axis=0)
    mean_W = np.mean(matrices, axis=0)    # Average connectivity matrix across seeds

    # Statistics of the averaged connectivity matrix
    mean_W_mean_weights = np.mean(mean_W)
    mean_W_std_weights = np.std(mean_W, ddof=1)

    return mean_W, mean_W_mean_weights, mean_W_std_weights


