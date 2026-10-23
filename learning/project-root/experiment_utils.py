
"""Utilities for loading and analyzing saved experiments."""

from pathlib import Path
from itertools import product
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd


def find_experiment(name, start_dir=None):
    start = Path(start_dir or Path.cwd()).resolve()
    for parent in (start, *start.parents):
        for directory in (parent / "results" / name, parent / name, parent):
            if (directory / "experiment.json").is_file() and directory.name == name:
                return directory
    raise FileNotFoundError(f"Experiment '{name}' not found. Specify its directory explicitly.")


class Experiment:
    """Access saved experiment metadata, conditions, and file paths."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        with open(self.directory / "experiment.json", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.conditions = pd.read_csv(self.directory / "conditions.csv")
        self.name = self.cfg.get("sim_name", self.directory.name)

    @classmethod
    def load(cls, directory):
        return cls(directory)

    @classmethod
    def find(cls, name, start_dir=None):
        return cls(find_experiment(name, start_dir))

    def __getattr__(self, name):
        if name in self.cfg:
            return self.cfg[name]
        for section in ("derived", "random_seeds"):
            if name in self.cfg.get(section, {}):
                return self.cfg[section][name]
        raise AttributeError(f"Experiment has no attribute '{name}'")

    def iter_params(self, *names, **filters):
        """Iterate over selected experiment levels with optional filters."""
        levels = {
            "condition": self.conditions["condition_id"].tolist(),
            "sim": range(1, len(self.configs) + 1),
            "pqif": self.pqif_values,
            "seed": range(self.cant_seed),
        }
        if not names:
            names = tuple(levels)

        unknown = set(names) - set(levels)
        if unknown:
            raise ValueError(f"Unknown iteration levels: {sorted(unknown)}")
        valid_filters = set(levels) | set(self.conditions.columns)
        unknown = set(filters) - valid_filters
        if unknown:
            raise ValueError(f"Unknown filters: {sorted(unknown)}")

        selected = self.conditions.copy()
        for key, value in filters.items():
            if key == "condition":
                selected = selected[selected["condition_id"].isin(np.atleast_1d(value))]
            elif key in selected.columns:
                selected = selected[selected[key].isin(np.atleast_1d(value))]

        levels["condition"] = selected["condition_id"].tolist()
        for key in ("sim", "pqif", "seed"):
            if key in filters:
                levels[key] = [v for v in levels[key] if v in np.atleast_1d(filters[key])]

        for values in product(*(levels[name] for name in names)):
            result = dict(zip(names, values))
            if "condition" in result:
                row = selected.loc[selected["condition_id"] == result["condition"]].iloc[0]
                result["condition"] = SimpleNamespace(
                    id=int(row["condition_id"]),
                    parameters=row.drop("condition_id").to_dict()
                )
            yield SimpleNamespace(**result)

    def path(self, name, condition=None, sim=None, pqif=None, seed=None, iloop=None):
        """Construct a path using the conventions in storage.py."""
        condition = getattr(condition, "id", condition)
        root = self.directory
        cdir = root / f"condition_{int(condition):03d}" if condition is not None else None
        sdir = cdir / f"simulation_{int(sim)}" if cdir is not None and sim is not None else None

        paths = {
            "base": root,
            "metadata": root / "experiment.json",
            "conditions": root / "conditions.csv",
            "condition": cdir,
            "target": cdir / "target_values.npy" if cdir else None,
            "target_parameters": cdir / "target_parameters.csv" if cdir else None,
            "target_time": cdir / "target_time.npy" if cdir else None,
            "fi_matching": cdir / "fi_matching_summary.csv" if cdir else None,
            "simulation": sdir,
            "connectivity": sdir / "connectivity_matrix" if sdir else None,
            "inputs": sdir / "inputs" if sdir else None,
            "outputs": sdir / "outputs" if sdir else None,
            "nspikes": sdir / "nspikes" if sdir else None,
            "currents": sdir / "currents" if sdir else None,
            "sequence_assignments": sdir / "sequence_assignments" if sdir else None,
        }

        if name == "external_current":
            if pqif is None:
                raise ValueError("external_current requires pqif")
            return root / f"external_current_pqif_{pqif}.npy"

        if name == "sequence_assignment":
            if sdir is None or pqif is None or seed is None:
                raise ValueError("sequence_assignment requires condition, sim, pqif, seed")
            return paths["sequence_assignments"] / f"sequence_sim_{sim}_pqif_{pqif}_seed_{seed}.csv"

        if name in ("inputs", "outputs", "nspikes", "connectivity_matrix"):
            if sdir is None or pqif is None or seed is None or iloop is None:
                raise ValueError(f"{name} requires condition, sim, pqif, seed, iloop")
            if name == "connectivity_matrix":
                return paths["connectivity"] / f"simulation_{sim}_connectivity_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npy"
            return paths[name] / f"simulation_{sim}_{name}_pqif_{pqif}_iloop_{iloop}_seed_{seed}.npz"

        if name not in paths:
            raise ValueError(f"Unknown path type: {name}")
        if paths[name] is None:
            raise ValueError(f"Path '{name}' requires condition and/or sim")
        return paths[name]

    def get_vreset(self, condition, sim=1):
        """Read the matched reset voltage for a condition and simulation."""
        condition_id = getattr(condition, "id", condition)
        path = self.path("fi_matching", condition=condition_id)
        if path.is_file():
            row = pd.read_csv(path).iloc[0]
            if sim == 1 and "best_vreset_overlap" in row:
                return float(row["best_vreset_overlap"])
            if sim == 2 and "best_vreset_gain" in row:
                return float(row["best_vreset_gain"])
        return float(self.configs[sim - 1]["vreset"])

    @staticmethod
    def order_neurons(activity, sequence_position):
        """Sort neurons by their sequence positions."""
        return activity[np.argsort(sequence_position), :]


    def load_target(
        self, condition, sim=None, pqif=None, seed=None,
        reorder=True, include_itstim=False
    ):
        """Load target activity, shaped (neurons, time)."""

        target = np.load(self.path("target", condition=condition))
        parameters = pd.read_csv(
            self.path("target_parameters", condition=condition)
        )

        if self.target_dynamics == "sequences" and not self.shared_sequence_order:
            if sim is None or pqif is None or seed is None:
                raise ValueError("Seed-specific targets require sim, pqif, and seed")

            canonical_order = np.argsort(
                parameters["sequence_position"].to_numpy(dtype=int)
            )

            parameters = pd.read_csv(
                self.path(
                    "sequence_assignment", condition=condition,
                    sim=sim, pqif=pqif, seed=seed
                )
            )

            positions = parameters["sequence_position"].to_numpy(dtype=int)
            target = target[canonical_order][positions]

        if reorder and "sequence_position" in parameters:
            target = self.order_neurons(
                target, parameters["sequence_position"].to_numpy(dtype=int)
            )

        if not include_itstim:
            target = target[:, self.itstim:]

        return target, parameters


    

    def load_network(
        self, condition, sim, pqif, seed,
        iloop=None, reorder=True, include_itstim=False
    ):
        """Load recurrent inputs and outputs, shaped (neurons, time)."""

        if iloop is None:
            iloop = self.nloop_train + 1

        order = None

        if reorder and self.target_dynamics == "sequences":
            _, parameters = self.load_target(
                condition, sim, pqif, seed,
                reorder=False, include_itstim=True
            )
            order = np.argsort(
                parameters["sequence_position"].to_numpy(dtype=int)
            )

        activities = []

        for name in ("inputs", "outputs"):
            path = self.path(
                name, condition=condition, sim=sim,
                pqif=pqif, seed=seed, iloop=iloop
            )

            with np.load(path) as data:
                activity = data[name].copy()

            if activity.shape == (self.itmax, self.N):
                activity = activity.T
            elif activity.shape != (self.N, self.itmax):
                raise ValueError(f"Unexpected activity shape: {activity.shape}")

            if order is not None:
                activity = activity[order]

            if not include_itstim:
                activity = activity[:, self.itstim:]

            activities.append(activity)

        return tuple(activities)
