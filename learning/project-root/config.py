
"""Central configuration for network training experiments.

Edit the defaults in Config, then create one instance with cfg = Config().
No files are written when this module is imported.

Values here are defaults but can be overwritten
"""

from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement, product
from typing import Any
import numpy as np


@dataclass
class Config:
    
    # Experiment
    
    master_seed: int = 2102
    target_dynamics: str = "sequences"  # "sequences" or "oscillations"

    
    # Network
    
    N: int = 200
    pqif_values: list[float] = field(default_factory=lambda: [0, 0.5, 1])
    configs: list[dict[str, float]] = field(
        default_factory=lambda: [{"vt": 0, "vreset": -21}]
    )
    p: float = 0.3
    gsyn: float = 0.5
    alpha: float = 0.25

    
    # Dynamics
    
    dt: float = 0.1
    itmax: int = 1000
    sigman: float = 1.0
    ts: float = 5.0

    
    # Stimulus + asymmetric stimulus interval
    
    itstim: int = 200
    current_amplitude: float = 5.0
    current_amplitude_by_pqif: dict[float, float] | None = None
    IEXT_LOW_FRAC: float = 0.0 
    IEXT_HIGH_FRAC: float = 1.0
    

    
    # Training
    
    nloop: int = 16
    nloop_train: int = 10
    cant_seed: int = 20
    ftrain: float = 1.0

    
    # Oscillation targets
    
    amp0: float = 4.0
    frequency_values: list[float] = field(default_factory=lambda: [1, 3, 5])

    
    # Sequence targets
    
    number_sequences: int = 1
    sequence_amplitudes: list[float] = field(
        default_factory=lambda: [4.0, 8.0, 12.0, 16.0]
    )
    sequence_sigmas: list[float] = field(
        default_factory=lambda: [12.5, 15.0, 17.5, 20.0]
    )
    sequence_offsets: list[float] = field(
        default_factory=lambda: [0.0, 5.0, 10.0]
    )

    sequence_ordering: str = "random"  # "random", "ordered", "alternating"
    shared_sequence_order: bool = True  # whether sequence orders are shared across seeds, default is true

    
    # Parallelization (only one level > 1 at a time)
    
    parallelize_seed: int = 1
    parallelize_pqif: int = 1
    parallelize_target_conditions: int = 50

    
    # Saving
    
    save_inputs: bool = True
    save_outputs: bool = True
    save_nspikes: bool = False
    save_dw: bool = False

    # Optional string with notes
    notes: str = "No notes added."

    
    # Derived values (not independently configurable)
    
    @property
    def N2(self) -> int:
        return self.N // 2

    @property
    def b(self) -> float:
        return 1.0 / self.ts

    @property
    def sequence_duration(self) -> float:
        return self.dt * self.itmax / self.number_sequences

    @property
    def sim_name(self) -> str:
        return f"{self.master_seed}_{self.target_dynamics}"

    @property
    def save_activity_loops(self) -> list[int]:
        return [0, self.nloop_train + 1, self.nloop - 1]

    @property
    def target_seed(self) -> int:
        return self.master_seed

    @property
    def external_current_seed(self) -> int:
        return self.master_seed + 100_000

    def network_seed(self, seed: int) -> int:
        """Preserve the original network seed convention for replicates."""
        return self.master_seed + seed + 1

    def target_conditions(self) -> list[dict[str, Any]]:
        """Enumerate active target conditions in deterministic order."""
        if self.target_dynamics == "sequences":
            return [
                {
                    "condition_id": i,
                    "amplitude": float(amplitude),
                    "sigma": float(sigma),
                    "offset": float(offset),
                }
                for i, (amplitude, sigma, offset) in enumerate(
                    product(
                        self.sequence_amplitudes,
                        self.sequence_sigmas,
                        self.sequence_offsets,
                    )
                )
            ]
        if self.target_dynamics == "oscillations":
            return [
                {"condition_id": i, "r1": float(r1), "r2": float(r2)}
                for i, (r1, r2) in enumerate(
                    combinations_with_replacement(self.frequency_values, 2)
                )
            ]
        raise ValueError(f"Unknown target_dynamics: {self.target_dynamics!r}")

    def sequence_seed(self,condition_id: int,num_simulation: int,pqif: float,seed: int,) -> np.random.SeedSequence:
        """When the ordering of neurons in the sequence is not shared across seeds, this function is used to handle reproducible randomization and independent of network initializatin."""
        return np.random.SeedSequence([
            self.master_seed,
            condition_id,
            num_simulation,
            int(round(pqif * self.N)),
            seed,
            300_000,
        ])

    def validate(self) -> None:
        """Fail early on inconsistent experiment settings."""
        if self.target_dynamics not in {"sequences", "oscillations"}:
            raise ValueError("target_dynamics must be 'sequences' or 'oscillations'")
        if not isinstance(self.master_seed, int) or self.master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if self.N <= 0 or self.N % 2 != 0:
            raise ValueError("N must be a positive even integer (oscillation targets split N in half)")
        if not 0 < self.p <= 1:
            raise ValueError("p must be in (0, 1]")
        if self.alpha <= 0 or self.ts <= 0 or self.dt <= 0:
            raise ValueError("alpha, ts and dt must be positive")
        if self.itmax <= 0 or not 0 <= self.itstim <= self.itmax:
            raise ValueError("Require itmax > 0 and 0 <= itstim <= itmax")
        if self.sigman < 0:
            raise ValueError("sigman must be nonnegative")
        if self.nloop < 2 or not 0 <= self.nloop_train <= self.nloop - 2:
            raise ValueError("Require nloop >= 2 and 0 <= nloop_train <= nloop - 2")
        if self.cant_seed < 1:
            raise ValueError("cant_seed must be >= 1")
        if not 0 < self.ftrain <= 1:
            raise ValueError("ftrain must be in (0, 1]")
        if not self.pqif_values or any(not 0 <= q <= 1 for q in self.pqif_values):
            raise ValueError("pqif_values must be nonempty and between 0 and 1")
        if not self.configs or any(set(c) != {"vt", "vreset"} for c in self.configs):
            raise ValueError("Each entry in configs must have 'vt' and 'vreset'")
        if self.number_sequences < 1:
            raise ValueError("number_sequences must be >= 1")
        if self.target_dynamics == "sequences":
            if not all((self.sequence_amplitudes, self.sequence_sigmas, self.sequence_offsets)):
                raise ValueError("Sequence parameter lists must be nonempty")
            if any(s <= 0 for s in self.sequence_sigmas):
                raise ValueError("Sequence sigmas must be positive")
        if self.target_dynamics == "oscillations" and not self.frequency_values:
            raise ValueError("frequency_values must be nonempty")
        if self.current_amplitude_by_pqif is not None:
            if set(self.current_amplitude_by_pqif) != set(self.pqif_values):
                raise ValueError("Specify current_amplitude for every pqif value")
        if self.sequence_ordering not in {"random", "ordered", "alternating"}:
            raise ValueError("Invalid sequence_ordering")
        if self.IEXT_LOW_FRAC >= self.IEXT_HIGH_FRAC:
            raise ValueError("IEXT_LOW_FRAC must be smaller than IEXT_HIGH_FRAC")
        workers = (
            self.parallelize_seed,
            self.parallelize_pqif,
            self.parallelize_target_conditions,
        )
        if any(not isinstance(n, int) or n < 1 for n in workers):
            raise ValueError("Parallelization settings must be positive integers")
        if sum(n > 1 for n in workers) > 1:
            raise ValueError("Parallelize at only one level: seed, pqif or target conditions")
        if not isinstance(self.shared_sequence_order, bool):
            raise ValueError("shared_sequence_order must be a boolean")

    def to_metadata(self) -> dict[str, Any]:
        """JSON-serializable snapshot of all parameters and derived settings."""
        self.validate()
        metadata = asdict(self)
        metadata["sim_name"] = self.sim_name
        metadata["derived"] = {
            "N2": self.N2,
            "b": self.b,
            "sequence_duration": self.sequence_duration,
            "save_activity_loops": self.save_activity_loops,
        }
        metadata["random_seeds"] = {
            "target_seed": self.target_seed,
            "external_current_seed": self.external_current_seed,
            "network_seed_rule": "master_seed + seed + 1",
        }
        return metadata
