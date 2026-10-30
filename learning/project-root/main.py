"""
Entry point for network training experiments.

Experiment level: one experiment per master_seed.
Condition level: Cartesian product of amplitude, sigma, and offset.
Simulation level: F-I-matched configurations for each condition.


Run with:
    python main.py
"""


from itertools import product
import time
import numpy as np
import pandas as pd
from config import Config
from experiment import run_experiment
from storage import initialize_experiment, experiment_logging
from fi_matching import find_optimal_vresets, save_fi_plot

def main():
    # debugging
    import matplotlib
    print("Active backend:", matplotlib.get_backend())
    # EXPERIMENT LEVEL
    experiment_settings = [
        {
            "master_seed": 3017,
            "N": 500,
            "dt": 0.1,
            "itmax": 1000,
            "nloop": 16,
            "nloop_train": 10,
            "cant_seed": 50,
            "p": 0.3,
            "sigman": 0.1, 
            "gsyn": 2.0,
            "alpha": 0.6,
            "learning_algorithm": "independent_rls",  # either shared_rls or independent_rls
            "allow_self_connections": False,
            "pqif_values": [0, 0.5, 1],
            "number_sequences": 1,
            "sequence_amplitudes": (np.linspace(2, 16, 3)).tolist(),
            "sequence_sigmas": (np.linspace(7, 11, 3)).tolist(),
            "sequence_offsets": (np.linspace(-10, 2, 3)).tolist(),
            "sequence_ordering": "random",
            "shared_sequence_order": True, 
            "current_amplitude": 16,
            "IEXT_LOW_FRAC": 0.0,
            "IEXT_HIGH_FRAC": 1.0,
            "parallelize_target_conditions": 1,
            "parallelize_pqif": 1,
            "parallelize_seed": 10,
            "notes": "Higher alpha and more negative offset",
        },
    ]

    # SIMULATION LEVEL
    use_fi_matching = True
    matching_rules = ["overlap"]
    manual_configs = [
        {"vt": 0.0, "vreset": -21.0},
        {"vt": 0.0, "vreset": -17.0},
    ]
    fi_settings = {
        "vreset_values": np.linspace(-35, -5, 61),
        "vt": 0.0,
        "dt": 0.1,
        "n_steps": 1000,
        "sigman": 0.0,
        "n_currents": 50,
        "fit_current_min": 0.7,
    }

    # EXPERIMENT LEVEL: PREPARE
    experiments = []
    for settings in experiment_settings:
        cfg_kwargs = dict(settings)
        cfg_kwargs["target_dynamics"] = "sequences"
        cfg_kwargs["configs"] = manual_configs[:len(matching_rules)] if use_fi_matching else manual_configs
        cfg = Config(**cfg_kwargs)
        cfg.validate()
        experiments.append(cfg)
        n_conditions = (
            len(cfg.sequence_amplitudes)
            * len(cfg.sequence_sigmas)
            * len(cfg.sequence_offsets)
        )
        print(f"Prepared {cfg.sim_name}: {n_conditions} conditions, ordering={cfg.sequence_ordering}")

    experiment_names = [cfg.sim_name for cfg in experiments]
    if len(experiment_names) != len(set(experiment_names)):
        raise ValueError("Duplicate experiment names. Assign distinct master_seed values.")

    # CONDITION LEVEL: PREPARE FI MATCHING
    fi_cache = {}
    for i, cfg in enumerate(experiments, start=1):
        experiment_dir = initialize_experiment(cfg)
        condition_configs = {}

        for condition_id, (amplitude, sigma, offset) in enumerate(product(
            cfg.sequence_amplitudes,
            cfg.sequence_sigmas,
            cfg.sequence_offsets,
        )):
            condition_dir = experiment_dir / f"condition_{condition_id:03d}"
            condition_dir.mkdir(parents=True, exist_ok=True)

            if use_fi_matching:
                key = (amplitude, offset)
                if key not in fi_cache:
                    fi_cache[key] = find_optimal_vresets(
                        amplitude=amplitude,
                        offset=offset,
                        **fi_settings,
                    )
                fi_result = fi_cache[key]
                configs = [
                    {
                        "vt": fi_settings["vt"],
                        "vreset": fi_result[f"best_vreset_{rule}"],
                    }
                    for rule in matching_rules
                ]
                fi_row = {
                    "amplitude": amplitude,
                    "sigma": sigma,
                    "offset": offset,
                    **fi_result,
                }
                pd.DataFrame([fi_row]).to_csv(
                    condition_dir / "fi_matching_summary.csv", index=False
                )
                save_fi_plot(
                    fi_result=fi_row,
                    configs=configs,
                    labels=matching_rules,
                    fi_settings=fi_settings,
                    save_path=condition_dir / "fi_matching_curves.svg",
                )
            else:
                configs = manual_configs

            condition_configs[condition_id] = configs
            print(f"Condition {condition_id:03d}: A={amplitude:g}, sigma={sigma:g}, offset={offset:g}")

        # EXPERIMENT LEVEL: RUN
        with experiment_logging(experiment_dir):
            print(f"Experiment {i}/{len(experiments)}: {cfg.sim_name}")
            run_experiment(
                cfg=cfg,
                experiment_dir=experiment_dir,
                condition_configs=condition_configs,
            )
            print(f"Completed experiment: {cfg.sim_name}")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    minutes, seconds = divmod(time.perf_counter() - start, 60)
    print(f"Execution time: {int(minutes)}m {seconds:.2f}s")