
"""
Storage utilities for network simulation experiments.

Responsibilities:
    - Create experiment and simulation directories
    - Save experiment metadata and condition mappings
    - Save targets and external currents
    - Save neuronal activity and connectivity
    - Save supplementary simulation results
    - Configure logging to terminal and file

This module does not generate targets or run simulations.
"""

import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd



# Experiment initialization


def initialize_experiment(cfg):
    """
    Create the root directory for a new experiment.

    Aborts if the experiment already exists, preventing overwriting.
    """

    project_dir = Path(__file__).resolve().parent
    experiment_dir = project_dir / "results" / cfg.sim_name

    cfg.validate()

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_dir}\n"
            "Aborting to prevent overwriting existing results."
        )

    experiment_dir.mkdir(parents=True)

    return experiment_dir


def save_metadata(cfg, experiment_dir):
    """
    Save a complete snapshot of the experiment configuration.

    All fields in Config are included automatically, together with
    derived parameters and random seed conventions.
    """

    metadata = cfg.to_metadata()

    path = Path(experiment_dir) / "experiment.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return path


def save_conditions(conditions, experiment_dir):
    """
    Save the mapping between condition IDs and target parameters.

    Works for both sequences and oscillations.
    """

    path = Path(experiment_dir) / "conditions.csv"

    df = pd.DataFrame(conditions)
    df.to_csv(path, index=False)

    return path



# Directory organization

def create_condition_directory(experiment_dir, condition_id):
    """
    Create the directory for one target condition.
    """
    path = (Path(experiment_dir) / f"condition_{condition_id:03d}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_simulation_directories(condition_dir, num_simulation):
    """
    Create the output directories for one network configuration.

    Returns:
        simulation_dir: Path to simulation directory
        folders: Dictionary of output subdirectories
    """

    simulation_dir = (Path(condition_dir) / f"simulation_{num_simulation}")

    folders = {
        "connectivity": simulation_dir / "connectivity_matrix",
        "currents": simulation_dir / "currents",
        "inputs": simulation_dir / "inputs",
        "outputs": simulation_dir / "outputs",
        "nspikes": simulation_dir / "nspikes",
    }

    simulation_dir.mkdir(parents=True, exist_ok=True)

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return simulation_dir, folders



# Target saving


def save_target(
    target,
    target_parameters,
    condition_dir,
    target_time=None,
):
    """
    Save target activity and its neuron-level parameters.

    Parameters
    ----------
    target : ndarray
        Target activity, shape (N, itmax).

    target_parameters : pandas.DataFrame
        Neuron-level target information.

    condition_dir : Path
        Directory for this target condition.

    target_time : ndarray, optional
        Time axis, used for sequence targets.
    """

    condition_dir = Path(condition_dir)
    np.save(condition_dir / "target_values.npy", np.asarray(target, dtype=np.float32),)
    target_parameters.to_csv(condition_dir / "target_parameters.csv", index=False,)

    if target_time is not None:
        np.save(condition_dir / "target_time.npy", np.asarray(target_time, dtype=np.float32),)


def save_external_current(Iext, experiment_dir):
    """
    Save the external current shared across conditions.
    """

    path = Path(experiment_dir) / "external_current.npy"
    np.save(path, np.asarray(Iext, dtype=np.float32))
    return path



# Network information


def save_neuron_assignment(pqif, N, simulation_dir):
    """
    Save which neurons use QIF or LIF dynamics.

    QIF neurons occupy the first int(N * pqif) indices,
    matching the simulation implementation.
    """

    nqif = int(N * pqif)

    neuron_type = np.full(N, "LIF", dtype=object)
    neuron_type[:nqif] = "QIF"

    df = pd.DataFrame({
        "neuron_index": np.arange(N),
        "neuron_model": neuron_type,
    })

    path = (Path(simulation_dir) / f"neuron_assignment_pqif_{pqif}.csv")

    df.to_csv(path, index=False)

    return path


def save_connectivity(
    w,
    num_simulation,
    pqif,
    seed,
    iloop,
    folders,
):
    """
    Save a network connectivity matrix for a specific loop.
    """

    path = (
        folders["connectivity"]
        / (
            f"simulation_{num_simulation}_connectivity_"
            f"pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy"
        )
    )

    np.save(path, np.asarray(w, dtype=np.float32))

    return path



# Activity saving


def save_activity(
    activity,
    activity_type,
    num_simulation,
    pqif,
    seed,
    iloop,
    folders,
):
    """
    Save one neuronal activity array in compressed format.

    Supported activity types:
        inputs
        outputs
        nspikes

    Continuous arrays are stored as float32.
    Spike arrays are stored as uint8.

    The activity type is also the key inside the .npz file.
    """

    if activity_type not in {"inputs", "outputs", "nspikes"}:
        raise ValueError(f"Unknown activity type: {activity_type}")

    dtype = (
        np.uint8 if activity_type == "nspikes"
        else np.float32
    )

    path = (
        folders[activity_type]
        / (
            f"simulation_{num_simulation}_{activity_type}_"
            f"pqif_{pqif}_iloop_{iloop}_seed_{seed}.npz"
        )
    )

    np.savez_compressed(path, **{activity_type: np.asarray(activity, dtype=dtype)},)

    return path



# Supplementary results


def save_dw(
    dw_buffer,
    num_simulation,
    pqif,
    seed,
    simulation_dir,
):
    """
    Save weight-change tracking data.

    dw_buffer contains (modt, modw) pairs.
    """

    if not dw_buffer:
        return None

    dw_array = np.asarray(dw_buffer)

    path = (
        Path(simulation_dir)
        / (
            f"simulation_{num_simulation}_dw_"
            f"pqif_{pqif}_seed_{seed}.npz"
        )
    )

    np.savez_compressed(
        path,
        modt=dw_array[:, 0].astype(np.int32),
        modw=dw_array[:, 1].astype(np.float32),
    )

    return path


def save_currents(
    currents_buffer,
    num_simulation,
    pqif,
    folders,
):
    """
    Save sampled external and recurrent currents.
    One CSV file per pqif condition, containing all seeds.
    """

    path = (
        folders["currents"]
        / f"simulation_{num_simulation}_currents_pqif_{pqif}.csv"
    )

    columns = [
        "pqif", "seed", "iloop", "it",
        "II_0", "v_0",
        "II_1", "v_1",
        "II_N2+1", "v_N2+1",
        "II_N2+2", "v_N2+2",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(currents_buffer)

    return path



# Logging


class Tee:
    """
    Write output to multiple streams simultaneously.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def experiment_logging(experiment_dir):
    """
    Redirect stdout and stderr to both terminal and log file.

    Automatically restores the original streams afterward.
    """

    log_path = Path(experiment_dir) / "experiment.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:

        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            yield log_path

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
