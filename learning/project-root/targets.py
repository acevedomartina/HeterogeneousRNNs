
"""
Generate target activity patterns for network training.
"""

import numpy as np
import pandas as pd


def generate_oscillations_target(r1, r2, cfg, rng):
    """
    Generate oscillatory target activity.

    Returns
    -------
    target : ndarray, shape (N, itmax)
    parameters : pandas.DataFrame
    """

    N = cfg.N
    itmax = cfg.itmax

    amplitude = rng.uniform(size=N) * cfg.amp0
    phase = rng.uniform(0, 2 * np.pi, size=N)

    indices = rng.permutation(N)

    frequencies = np.zeros(N)
    frequencies[indices[:cfg.N2]] = r1
    frequencies[indices[cfg.N2:]] = r2

    omega = frequencies * 2 * np.pi / itmax
    time_steps = np.arange(itmax)

    target = amplitude[:, None] * np.cos(
        omega[:, None] * time_steps[None, :]
        + phase[:, None]
    )

    parameters = pd.DataFrame({
        "neuron_index": np.arange(N),
        "frequency_pair": [f"({r1},{r2})"] * N,
        "romega_assigned": frequencies,
        "omega_per_timestep": omega,
        "phase": phase,
        "amplitude": amplitude,
        "amp0": np.full(N, cfg.amp0),
    })

    return target, parameters



def generate_sequences_target(amplitude, sigma, offset, cfg, rng):
    """
    Generate periodic sequential activity.

    Sequence ordering options:
        random: Randomized neuron-to-position assignment.
        ordered: Neurons follow their original indices.
        alternating: First and second halves alternate in sequence order.

    The same target is used for all pqif values.
    """

    N = cfg.N
    itmax = cfg.itmax
    dt = cfg.dt

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if cfg.number_sequences < 1:
        raise ValueError("number_sequences must be >= 1")

    sequence_duration = dt * itmax / cfg.number_sequences
    target_time = np.arange(itmax) * dt
    peak_times_by_position = np.arange(N) * sequence_duration / N

    # SEQUENCE ORDERING
    if cfg.sequence_ordering == "random":
        neuron_permutation = rng.permutation(N)
    elif cfg.sequence_ordering == "ordered":
        neuron_permutation = np.arange(N)
    elif cfg.sequence_ordering == "alternating":
        if N % 2 != 0:
            raise ValueError("Alternating ordering requires even N")
        neuron_permutation = np.empty(N, dtype=int)
        neuron_permutation[:N // 2] = np.arange(0, N, 2)
        neuron_permutation[N // 2:] = np.arange(1, N, 2)
    else:
        raise ValueError(f"Unknown sequence_ordering: {cfg.sequence_ordering}")

    peak_times_by_neuron = peak_times_by_position[neuron_permutation]

    # PERIODIC GAUSSIAN SEQUENCES
    time_distance = (
        (target_time[None, :] - peak_times_by_neuron[:, None] + sequence_duration / 2)
        % sequence_duration - sequence_duration / 2
    )

    target = offset + amplitude * np.exp(
        -(time_distance ** 2) / (2 * sigma ** 2)
    )

    parameters = pd.DataFrame({
        "neuron_index": np.arange(N),
        "sequence_position": neuron_permutation,
        "peak_time": peak_times_by_neuron,
        "amplitude": np.full(N, amplitude),
        "sigma": np.full(N, sigma),
        "offset": np.full(N, offset),
        "sequence_duration": np.full(N, sequence_duration),
    })

    return target, parameters, target_time


def generate_target(condition, cfg, rng):
    """
    Generate a target using the selected target dynamics.
    """

    if cfg.target_dynamics == "oscillations":

        r1 = condition["r1"]
        r2 = condition["r2"]

        target, parameters = generate_oscillations_target(
            r1=r1,
            r2=r2,
            cfg=cfg,
            rng=rng,
        )

        return target, parameters, None

    elif cfg.target_dynamics == "sequences":

        return generate_sequences_target(
            amplitude=condition["amplitude"],
            sigma=condition["sigma"],
            offset=condition["offset"],
            cfg=cfg,
            rng=rng,
        )

    else:
        raise ValueError(
            f"Unknown target dynamics: {cfg.target_dynamics}"
        )


def order_sequence_target(target, parameters):
    """
    Reorder target rows into temporal sequence order.

    This applies the inverse of the original neuron permutation.

    The original target is not modified.
    """

    sequence_position = parameters["sequence_position"].to_numpy(dtype=int)

    # Inverse permutation:
    # ordered neuron index for each sequence position
    inverse_permutation = np.argsort(sequence_position)

    return target[inverse_permutation, :]
