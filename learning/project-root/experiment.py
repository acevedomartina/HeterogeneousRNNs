
"""
Handles running the experiments and parallell execution.
In main.py, we can choose what to parallelize over.
"""

import copy
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from simulation import (
    run_single_seed,
    currents,
)
from targets import generate_target
from storage import (
    create_condition_directory,
    create_simulation_directories,
    save_conditions,
    save_currents,
    save_metadata,
    save_neuron_assignment,
    save_target,
)


# PARALLEL EXECUTION

def run_parallel(function, items, n_jobs):

    if n_jobs == 1:
        return [function(item) for item in items]

    return Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(function)(item)
        for item in items
    )


# ONE SEED

def run_seed(
    seed,
    pqif,
    num_simulation,
    simulation_config,
    target,
    Iext,
    cfg,
    folders,
):

    return run_single_seed(
        seed=seed,
        pqif=pqif,
        num_simulation=num_simulation,
        simulation_config=simulation_config,
        target=target,
        Iext=Iext,
        cfg=cfg,
        folders=folders,
    )


# ONE PQIF

def run_pqif(
    pqif,
    num_simulation,
    simulation_config,
    target,
    Iext,
    cfg,
    folders,
    condition,  # is for when target is not shared across seeds
):
    """Runs simulation for one pqif, which may have multiple seeds"""

    print(f"    Running pqif={pqif}, \nseeds={cfg.cant_seed}")

    simulation_dir = folders["connectivity"].parent

    save_neuron_assignment(
        pqif=pqif,
        N=cfg.N,
        simulation_dir=simulation_dir,
    )

    def execute_seed(seed):

        """
        Executes one seeds. Handles specific rules for seeds.
        """

        # If the sequence order is not shared across seeds, make a new sequence rng per seed
        if (cfg.target_dynamics == "sequences" and not cfg.shared_sequence_order):

            sequence_rng = np.random.default_rng(
                cfg.sequence_seed(
                    condition_id=int(condition["condition_id"]),
                    num_simulation=num_simulation,
                    pqif=pqif,
                    seed=seed,
                )
            )

            seed_target, seed_parameters, seed_time = generate_target(
                condition=condition,
                cfg=cfg,
                rng=sequence_rng,
            )

            # Save the neuron-to-sequence assignment for this seed
            sequence_dir = folders["connectivity"].parent / "sequence_assignments"
            sequence_dir.mkdir(parents=True, exist_ok=True)

            seed_parameters.to_csv(
                sequence_dir
                / f"sequence_sim_{num_simulation}_pqif_{pqif}_seed_{seed}.csv",
                index=False,
            )

        else:
            # if not, use the target that was passed to the function
            seed_target = target

        return run_seed(
            seed=seed,
            pqif=pqif,
            num_simulation=num_simulation,
            simulation_config=simulation_config,
            target=seed_target,
            Iext=Iext[pqif],
            cfg=cfg,
            folders=folders,
        )

    seed_results = run_parallel(
        function=execute_seed,
        items=range(cfg.cant_seed),
        n_jobs=cfg.parallelize_seed,
    )

    all_currents = [
        row
        for seed_buffer in seed_results
        for row in seed_buffer
    ]

    save_currents(
        currents_buffer=all_currents,
        num_simulation=num_simulation,
        pqif=pqif,
        folders=folders,
    )

    return None


# ONE NETWORK CONFIGURATION

def run_simulation_config(
    num_simulation,
    simulation_config,
    target,
    Iext,
    cfg,
    condition_dir,
    condition,
):

    print(f"  Simulation {num_simulation}: \nvt={simulation_config['vt']}, \nvreset={simulation_config['vreset']}")

    simulation_dir, folders = create_simulation_directories(
        condition_dir=condition_dir,
        num_simulation=num_simulation,
    )

    def execute_pqif(pqif):
        return run_pqif(
            pqif=pqif,
            num_simulation=num_simulation,
            simulation_config=simulation_config,
            target=target,
            Iext=Iext,
            cfg=cfg,
            folders=folders,
            condition=condition
        )

    run_parallel(
        function=execute_pqif,
        items=cfg.pqif_values,
        n_jobs=cfg.parallelize_pqif,
    )

    return None


# ONE TARGET CONDITION

# CONDITION LEVEL

def run_condition(condition, cfg, experiment_dir, Iext, condition_configs):
    """
    We often do parameter sweeps, therefore we initialize targets from here
    (one condition per unique combination of the target parameters). Also,
    different parameters impact the input current range, therefore we also
    do the FI matching here
    """
    condition_id = int(condition["condition_id"])
    print(f"Condition {condition_id}: {condition}", flush=True)

    condition_dir = create_condition_directory(
        experiment_dir=experiment_dir,
        condition_id=condition_id,
    )

    # Below: Generates a target that is shared across seeds.
    # If not shared across seeds, it is handled further in (in run_pqif())

    target_rng = np.random.default_rng(cfg.target_seed)
    target, target_parameters, target_time = generate_target(
        condition=condition,
        cfg=cfg,
        rng=target_rng,
    )

    save_target(
        target=target,
        target_parameters=target_parameters,
        target_time=target_time,
        condition_dir=condition_dir,
    )

    # SIMULATION LEVEL: CONDITION-SPECIFIC F-I MATCHING
    for num_simulation, simulation_config in enumerate(
        condition_configs[condition_id], start=1
    ):
        run_simulation_config(
            num_simulation=num_simulation,
            simulation_config=simulation_config,
            target=target,
            Iext=Iext,
            cfg=cfg,
            condition_dir=condition_dir,
            condition=condition,
        )

    return condition_id


# FULL EXPERIMENT

# EXPERIMENT LEVEL

def run_experiment(cfg, experiment_dir, condition_configs):

    cfg.validate()  # in config.py, validate() ensures that we do not have "illegal" parameter values

    experiment_dir = Path(experiment_dir)

    # SAVE EXPERIMENT INFORMATION

    save_metadata(
        cfg=cfg,
        experiment_dir=experiment_dir,
    )

    conditions = cfg.target_conditions()

    save_conditions(
        conditions=conditions,
        experiment_dir=experiment_dir,
    )

    print("-" * 60)
    print(f"Experiment: {cfg.sim_name}")
    print(f"Target dynamics: {cfg.target_dynamics}")
    print(f"Conditions: {len(conditions)}")
    print(f"Network configurations: {len(cfg.configs)}")
    print(f"QIF fractions: {len(cfg.pqif_values)}")
    print(f"Seeds: {cfg.cant_seed}")
    print("-" * 60)

    # EXTERNAL CURRENT
    # There exist a choice to give different external currents to the different pqif, from
    # when we suspected this impact training performance. Choosing this is, as other stuff,
    # handled in main.py

    Iext = {}

    for pqif in cfg.pqif_values:

        current_amplitude = (
            cfg.current_amplitude
            if cfg.current_amplitude_by_pqif is None
            else cfg.current_amplitude_by_pqif[float(pqif)]
        )

        current_cfg = copy.copy(cfg)
        current_cfg.current_amplitude = current_amplitude

        current_rng = np.random.default_rng(
            cfg.external_current_seed
        )

        # Iext[pqif] = generate_external_current(
        #     cfg=current_cfg,
        #     rng=current_rng,
        # )

        Iext[pqif] = currents(            
            cfg=current_cfg,
            rng=current_rng,
        )

        np.save(experiment_dir / f"external_current_pqif_{pqif}.npy", Iext[pqif],)

        print(f"External current: pqif={pqif}, amplitude={current_amplitude}")

    # EXECUTE ALL TARGET CONDITIONS

    # CONDITION LEVEL: PARALLEL EXECUTION
    def execute_condition(condition):
        return run_condition(
            condition=condition,
            cfg=cfg,
            experiment_dir=experiment_dir,
            Iext=Iext,
            condition_configs=condition_configs,
        )

    completed_conditions = run_parallel(
        function=execute_condition,
        items=conditions,
        n_jobs=cfg.parallelize_target_conditions,
    )

    print("-" * 60)
    print("Experiment completed successfully.")
    print(
        f"Completed conditions: "
        f"{len(completed_conditions)}"
    )
    print(f"Results saved to: {experiment_dir}")
    print("-" * 60)

    return experiment_dir
