"""Run the clean 20-seed current/connectivity sweep."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SIMULATION_SCRIPT = SCRIPT_DIR / "run_simulations.py"
MASTER_SEED = int(os.environ.get("SWEEP_MASTER_SEED", "9400"))
CANT_SEED = 20
PARALLEL_SEED = int(os.environ.get("SWEEP_PARALLEL_SEED", "10"))
SIGMAN = 0.1
VRESET = -16.0
TARGET_AMPLITUDE = 8
AMPLITUDES = [4, 8, 12, 16, 20, 24]
GSYNS = [0.1, 0.5, 1, 2]
SWEEP_NAME = "seeds_20"
SWEEP_DIR = SCRIPT_DIR / SWEEP_NAME
MANIFEST_PATH = SWEEP_DIR / "manifest.csv"


def experiment_name(amplitude: int, gsyn: float) -> str:
    return f"{SWEEP_NAME}_ampc_{amplitude:g}_g{gsyn:g}"


def run_combination(amplitude: int, gsyn: float) -> tuple[str, int]:
    name = experiment_name(amplitude, gsyn)
    output_dir = SCRIPT_DIR / name
    if output_dir.exists():
        print(f"Reusing existing result {name}", flush=True)
        return name, 0

    environment = os.environ.copy()
    environment.update(
        {
            "MASTER_SEED": str(MASTER_SEED),
            "SIM_SIGMAN": str(SIGMAN),
            "SIM_CURRENT_AMPLITUDE": str(amplitude),
            "SIM_CURRENT_LOW_FRAC": "0",
            "SIM_CURRENT_HIGH_FRAC": "1",
            "SIM_VRESET": str(VRESET),
            "SIM_GSYN": str(gsyn),
            "SIM_TARGET_AMPLITUDE": str(TARGET_AMPLITUDE),
            "SIM_CANT_SEED": str(CANT_SEED),
            "SIM_PARALLEL_SEED": str(PARALLEL_SEED),
            "SIM_NAME": name,
        }
    )
    print(f"Starting {name}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(SIMULATION_SCRIPT)],
        cwd=SCRIPT_DIR,
        env=environment,
        check=False,
    )
    return name, completed.returncode


def main() -> int:
    if not SIMULATION_SCRIPT.exists():
        raise FileNotFoundError(SIMULATION_SCRIPT)
    SWEEP_DIR.mkdir(exist_ok=True)
    fieldnames = ["timestamp", "experiment", "amplitude", "gsyn", "status", "return_code"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for amplitude, gsyn in product(AMPLITUDES, GSYNS):
            name, return_code = run_combination(amplitude, gsyn)
            status = "completed" if return_code == 0 else "failed"
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experiment": name,
                    "amplitude": amplitude,
                    "gsyn": gsyn,
                    "status": status,
                    "return_code": return_code,
                }
            )
            file.flush()
            if return_code:
                print(f"Failed: {name}", file=sys.stderr)
                return return_code
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
