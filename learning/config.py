from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json


@dataclass(frozen=True)
class SimParam:
    sim: int
    vt: float
    vreset: float
    gain: float | None = None  # for now, we don't have an automated way to set gain values from vreset, we do it manually. This could be improved later


class Config:

    def __init__(self, metadata, base_path):
        self.metadata = metadata
        self.base_path = Path(base_path)

        self.name = metadata["name"]

        self.sims = [
            SimParam(**sim)
            for sim in metadata["simulations"]
        ]

        self.pqif_numbers = metadata["conditions"]["pqif_values"]
        self.seed_numbers = range(
            metadata["conditions"]["n_seeds"]
        )

        self._sim_lookup = {
            sim.sim: sim
            for sim in self.sims
        }

    def get_sim(self, sim):
        """Return information associated with a simulation number."""
        try:
            return self._sim_lookup[sim]
        except KeyError:
            raise ValueError(
                f"Unknown simulation {sim}. "
                f"Available simulations: {self.sim_numbers}"
            )

    @classmethod
    def from_file(cls, base_path):
        base_path = Path(base_path)

        with open(
            base_path / "experiment.json",
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        return cls(
            metadata=metadata,
            base_path=base_path,
        )

    @property
    def sim_numbers(self):
        return [s.sim for s in self.sims]

    @property
    def vresets(self):
        return [s.vreset for s in self.sims]

    @property
    def parameters(self):
        """All dimensions that can be iterated over."""
        return {
            "sim": self.sims,
            "pqif": self.pqif_numbers,
            "seed": self.seed_numbers,
        }

    @property
    def target(self):
        return self.metadata["target"]

    @property
    def network(self):
        return self.metadata["network"]

    @property
    def dynamics(self):
        return self.metadata["dynamics"]

    @property
    def stimulus(self):
        return self.metadata["stimulus"]

    @property
    def training(self):
        return self.metadata["training"]

    @property
    def target_dynamics(self):
        return self.metadata["target"]["type"]

    @property
    def master_seed(self):
        return self.metadata["master_seed"]

    def iter_params(self, *names):
        """
        Iterate over parameter combinations.

        With no arguments:
            iterate over everything.

        With arguments:
            iterate over only those dimensions.

        Examples
        --------
        config.iter_params()
        config.iter_params("sim")
        config.iter_params("sim", "pqif")
        config.iter_params("pqif", "seed")
        """
        params = self.parameters

        # No names means all dimensions
        if not names:
            names = tuple(params)

        unknown = set(names) - set(params)
        if unknown:
            raise ValueError(
                f"Unknown parameters: {sorted(unknown)}. "
                f"Available parameters: {list(params)}"
            )

        for values in product(*(params[name] for name in names)):
            yield dict(zip(names, values))


    # ------------------------------------------------------------------
    # Paths

    def path(self, name, sim=None):
        """Return a path belonging to this configuration."""

        base_path = self.base_path

        # Accept either SimParam or simulation number
        if isinstance(sim, SimParam):
            sim = sim.sim

        simulation_dir = (
            base_path / f"simulation_{sim}"
            if sim is not None
            else None
        )

        paths = {
            "base": base_path,
            "target": base_path / "target_values.npy",
            "target_parameters": base_path / "target_parameters.csv",
            "external_current": base_path / "external_current.npy",

            "simulation": simulation_dir,

            "connectivity": (
                simulation_dir / "connectivity_matrix"
                if simulation_dir is not None else None
            ),
            "outputs": (
                simulation_dir / "outputs"
                if simulation_dir is not None else None
            ),
            "inputs": (
                simulation_dir / "inputs"
                if simulation_dir is not None else None
            ),
            "currents": (
                simulation_dir / "currents"
                if simulation_dir is not None else None
            ),
            "nspikes": (
                simulation_dir / "nspikes"
                if simulation_dir is not None else None
            ),
            # "activity": (
            #     simulation_dir / "activity_examples"
            #     if simulation_dir is not None else None
            # ),
        }

        if name not in paths:
            raise ValueError(
                f"Unknown path {name!r}. "
                f"Choose one of: {', '.join(paths)}"
            )

        path = paths[name]

        if path is None:
            raise ValueError(
                f"Path {name!r} requires a simulation."
            )

        return path

    def __repr__(self):
        sims = ", ".join(
            f"{s.sim}: vt={s.vt}, vreset={s.vreset}"
            for s in self.sims
        )

        return (
            f"Config name: {self.name}\n"
            f"  target: {self.target_dynamics}\n"
            f"  pqif: {self.pqif_numbers}\n"
            f"  simulations: {sims}\n"
            f"  seeds: {len(self.seed_numbers)}\n"
            f"  base path: {self.base_path}"
        )