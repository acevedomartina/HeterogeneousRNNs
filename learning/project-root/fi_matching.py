
"""
Find LIF reset voltages matching the QIF f-I curve or gain.
For difference sequence parameters, the input current range is quite different.
Therefore, we find the best match per sequence target in order to maintain that
LIF and QIF is as close as possible on the FI curve.

It also saves the FI curve plot for each time for reference.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_single_cell(current, neuron_type, vreset=0.0, vt=0.0,
                         dt=0.1, n_steps=1000, sigman=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = float(vreset)
    spikes = 0

    for _ in range(n_steps):
        noise = rng.normal() * sigman if sigman > 0 else 0.0

        if neuron_type == "LIF":
            dx = -x + current + noise
            x_new = x + dt * dx
            if x < vt and x_new > vt:
                spikes += 1
                x_new = vreset

        elif neuron_type == "QIF":
            dx = 1 - np.cos(x) + current * (1 + np.cos(x)) + noise
            x_new = x + dt * dx
            dpi = (np.pi - (x % (2 * np.pi))) % (2 * np.pi)
            if x_new - x > dpi:
                spikes += 1

        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        x = x_new

    return spikes


def fi_curve(currents, neuron_type, vreset=0.0, vt=0.0,
             dt=0.1, n_steps=1000, sigman=0.0):
    simulation_seconds = n_steps / 1000

    return np.array([
        simulate_single_cell(
            current, neuron_type, vreset=vreset, vt=vt,
            dt=dt, n_steps=n_steps, sigman=sigman, seed=i
        ) / simulation_seconds
        for i, current in enumerate(currents)
    ])


def fit_fi_gain(currents, rates, fit_min=0.7, fit_max=None, min_fit_points=5):
    mask = currents >= fit_min

    if fit_max is not None:
        mask &= currents <= fit_max

    if mask.sum() < max(min_fit_points, 3):
        raise ValueError("Not enough current points for the linear fit.")

    x = currents[mask]
    y = rates[mask]

    coef, cov = np.polyfit(x, y, 1, cov=True)
    predicted = np.polyval(coef, x)

    ss_res = np.sum((y - predicted)**2)
    ss_tot = np.sum((y - y.mean())**2)

    return {
        "gain": float(coef[0]),
        "gain_err": float(np.sqrt(max(cov[0, 0], 0))),
        "intercept": float(coef[1]),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def overlap_error(lif_rates, qif_rates, lif_fit, qif_fit):
    return float(np.mean((lif_rates - qif_rates)**2))


def gain_error(lif_rates, qif_rates, lif_fit, qif_fit):
    return float(abs(lif_fit["gain"] - qif_fit["gain"]))


comparison_rules = {
    "overlap": overlap_error,
    "gain": gain_error,
}


def find_optimal_vresets(
    amplitude,
    offset,
    vreset_values=None,
    vt=0.0,
    dt=0.1,
    n_steps=1000,
    sigman=0.0,
    n_currents=50,
    fit_current_min=0.7,
    fit_current_max=None,
    current_scale=1.0,
    current_shift=0.0,
    current_min_floor=0.0,
):
    """Find optimal LIF reset voltages over the target-derived current range."""

    if vreset_values is None:
        vreset_values = np.linspace(-35, -5, 61)

    current_min = max(
        current_min_floor, current_scale * offset + current_shift
    )
    current_max = current_scale * (offset + amplitude) + current_shift

    if current_max <= current_min:
        raise ValueError("The current interval must have positive width.")

    currents = np.linspace(current_min, current_max, n_currents)
    fit_min = max(current_min, fit_current_min)

    if fit_current_max is not None:
        fit_max = min(current_max, fit_current_max)
    else:
        fit_max = current_max

    if fit_min >= fit_max:
        raise ValueError("No valid current interval for fitting the gain.")

    simulation_args = {
        "vt": vt,
        "dt": dt,
        "n_steps": n_steps,
        "sigman": sigman,
    }

    qif_rates = fi_curve(currents, "QIF", **simulation_args)
    qif_fit = fit_fi_gain(
        currents, qif_rates, fit_min=fit_min, fit_max=fit_max
    )

    candidates = []

    for vreset in vreset_values:
        lif_rates = fi_curve(
            currents, "LIF", vreset=float(vreset), **simulation_args
        )
        lif_fit = fit_fi_gain(
            currents, lif_rates, fit_min=fit_min, fit_max=fit_max
        )

        errors = {
            name: rule(lif_rates, qif_rates, lif_fit, qif_fit)
            for name, rule in comparison_rules.items()
        }

        candidates.append({
            "vreset": float(vreset),
            "gain": lif_fit["gain"],
            "gain_err": lif_fit["gain_err"],
            "fit_r2": lif_fit["r2"],
            "errors": errors,
        })

    result = {
        "amplitude": float(amplitude),
        "offset": float(offset),
        "current_min": float(current_min),
        "current_max": float(current_max),
        "fit_min": float(fit_min),
        "fit_max": float(fit_max),
        "qif_gain": qif_fit["gain"],
        "qif_gain_err": qif_fit["gain_err"],
        "qif_fit_r2": qif_fit["r2"],
    }

    for name in comparison_rules:
        best = min(
            candidates,
            key=lambda item: item["errors"][name]
        )

        result[f"best_vreset_{name}"] = best["vreset"]
        result[f"lif_gain_{name}"] = best["gain"]
        result[f"lif_gain_err_{name}"] = best["gain_err"]
        result[f"lif_fit_r2_{name}"] = best["fit_r2"]
        result[f"error_{name}"] = best["errors"][name]

    return result


def save_fi_plot(fi_result, configs, labels, fi_settings, save_path):
    currents = np.linspace(
        fi_result["current_min"],
        fi_result["current_max"],
        fi_settings["n_currents"],
    )
    simulation_args = {
        "vt": fi_settings["vt"],
        "dt": fi_settings["dt"],
        "n_steps": fi_settings["n_steps"],
        "sigman": fi_settings["sigman"],
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    qif_rates = fi_curve(currents, "QIF", **simulation_args)
    ax.plot(currents, qif_rates, color="black", linewidth=2, label="QIF")

    for label, config in zip(labels, configs):
        lif_rates = fi_curve(
            currents, "LIF", vreset=config["vreset"],
            **{**simulation_args, "vt": config["vt"]},
        )
        ax.plot(
            currents, lif_rates, linewidth=1.8,
            label=f"LIF ({label}, Vreset={config['vreset']:.2f})",
        )

    ax.axvspan(
        fi_result["fit_min"], fi_result["fit_max"],
        color="gray", alpha=0.1, label="Gain fit interval",
    )
    ax.set_xlabel("Input current")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(
        f"f–I curves | A={fi_result['amplitude']:g}, "
        f"offset={fi_result['offset']:g}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
