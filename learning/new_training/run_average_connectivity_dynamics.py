"""Test the dynamics with connectivity averaged over the 20 training seeds."""

from __future__ import annotations
from snn_core import dynamics, detect, evolution, initialize_neurons, initialize_connectivity_matrix, initialize_training
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS AND PARAMETERS (only what defines the sweep; the rest comes from JSON)
# =============================================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "average_connectivity_20seeds"
CURRENT_AMPLITUDES = [2, 4, 8, 12, 16, 20]
GSYN_VALUES = [0.1, 0.5, 1, 2]
TEST_SEED = 78001


def experiment_dir(current_amplitude, gsyn):
    return SCRIPT_DIR / f"seeds_20_ampc_{current_amplitude:g}_g{gsyn:g}"


def load_params(exp_dir):
    """Read every simulation parameter from the experiment's experiment.json."""
    with open(exp_dir / "experiment.json", encoding="utf-8") as f:
        cfg = json.load(f)
    sim = cfg["simulations"][0]          # simulation_1 (the only one used here)
    prm = {
        **cfg["network"],                # N, p, gsyn, alpha
        **cfg["dynamics"],               # dt, itmax, sigman, ts, b
        **cfg["stimulus"],               # itstim, current_amplitude, current_low_frac, current_high_frac
        **cfg["training"],               # nloop, nloop_train, ftrain
        "pqif_values": cfg["conditions"]["pqif_values"],
        "n_seeds": cfg["conditions"]["n_seeds"],
        "vt": sim["vt"],
        "vreset": sim["vreset"],
        "master_seed": cfg["random_seeds"]["target_seed"],
        "target": cfg["target"],
    }
    prm["N2"] = prm["N"] // 2
    prm["iloop_target"] = prm["nloop_train"] + 1   # loop where the post-training matrix is saved
    return prm

def experiment_dir(current_amplitude, gsyn):
    return SCRIPT_DIR / f"seeds_20_ampc_{current_amplitude:g}_g{gsyn:g}"


# =============================================================================
# 3. INPUTS, TARGET, AND AVERAGED CONNECTIVITY
# =============================================================================

def load_external_current(exp_dir, N, itmax):
    Iext = np.load(exp_dir / "external_current.npy").astype(float)
    if Iext.shape == (itmax, N):
        Iext = Iext.T
    if Iext.shape != (N, itmax):
        raise ValueError(f"Unexpected external-current shape: {Iext.shape}")
    return Iext

def load_target(experiment_dir):
    return np.load(
        experiment_dir / "condition_000" / "target_values.npy"
    ).astype(float)


def get_average_matrix(exp_dir, pqif, n_seeds, iloop_target):
    conn_dir = exp_dir / "condition_000" / "simulation_1" / "connectivity_matrix"
    matrices = []
    for seed in range(n_seeds):
        path = (
            conn_dir
            / f"simulation_1_connectivity_pqif_{pqif}_iloop_{iloop_target}_seed_{seed}.npy"
        )
        matrices.append(np.asarray(np.load(path), dtype=float))
    return np.asmatrix(np.mean(matrices, axis=0))   # np.matrix, as evolution() expects


def calculate_r2(values, target, itstim, itmax):
    values = values[itstim:itmax]
    target = target[:, itstim:itmax].T
    denominator = np.sum((target - target.mean()) ** 2)
    return float(1 - np.sum((values - target) ** 2) / denominator)

# =============================================================================
# 4. RUN ALL PARAMETERS WITH THE AVERAGE MATRIX
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = []
    all_params = {}

    for current_amplitude in CURRENT_AMPLITUDES:
        for gsyn in GSYN_VALUES:
            exp_dir = experiment_dir(current_amplitude, gsyn)
            P = load_params(exp_dir)

            # Check that the folder name agrees with what the experiment actually used
            assert np.isclose(P["current_amplitude"], current_amplitude), exp_dir
            assert np.isclose(P["gsyn"], gsyn), exp_dir

            N, itmax, itstim = P["N"], P["itmax"], P["itstim"]
            Iext = load_external_current(exp_dir, N, itmax)
            target = load_target(exp_dir)
            iout = np.arange(N)

            test_dir = OUTPUT_DIR / f"ampc_{current_amplitude:g}_g{gsyn:g}"
            test_dir.mkdir(exist_ok=True)
            all_params[test_dir.name] = P

            for pqif in P["pqif_values"]:
                nqif = int(N * pqif)
                rng = np.random.default_rng(
                    TEST_SEED
                    + current_amplitude * 100
                    + int(gsyn * 10)
                    + int(pqif * 1000)
                )
                w_avg = get_average_matrix(exp_dir, pqif, P["n_seeds"], P["iloop_target"])
                x, r, _ = initialize_neurons(N, nqif, rng, P["vreset"], P["vt"])
                inputs_loop, outputs_loop = [], []
                start_time = time.time()

                for it in range(itmax):
                    nspike = np.zeros(N)
                    x, r, nspike, rout, II, v_syn = evolution(
                        x, r, Iext, w_avg, nqif, it, P["dt"], iout,
                        nspike, P["b"], P["vt"], P["vreset"], rng, N, P["sigman"],
                    )
                    inputs_loop.append(v_syn)
                    outputs_loop.append(rout)

                inputs_loop = np.asarray(inputs_loop, dtype=np.float32)
                outputs_loop = np.asarray(outputs_loop, dtype=np.float32)
                np.save(test_dir / f"w_average_used_pqif_{pqif:g}.npy", np.asarray(w_avg))
                np.savez_compressed(test_dir / f"test_inputs_pqif_{pqif:g}.npz", inputs=inputs_loop)
                np.savez_compressed(test_dir / f"test_outputs_pqif_{pqif:g}.npz", outputs=outputs_loop)

                summary.append({
                    "current_amplitude": current_amplitude,
                    "gsyn": gsyn,
                    "pqif": pqif,
                    "sigman": P["sigman"],
                    "vreset": P["vreset"],
                    "connectivity_iloop": P["iloop_target"],
                    "r2_recurrent_input": calculate_r2(inputs_loop, target, itstim, itmax),
                    "r2_output": calculate_r2(outputs_loop, target, itstim, itmax),
                    "elapsed_seconds": time.time() - start_time,
                })
                print(f"Completed ampc={current_amplitude:g}, gsyn={gsyn:g}, pqif={pqif:g}", flush=True)

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "average_dynamics_r2.csv", index=False)
    with (OUTPUT_DIR / "configuration.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "current_amplitudes": CURRENT_AMPLITUDES,
                "gsyn_values": GSYN_VALUES,
                "test_seed": TEST_SEED,
                "same_external_current": True,
                "parameters_per_experiment": all_params,
            },
            f, indent=2,
        )

if __name__ == "__main__":
    main()
