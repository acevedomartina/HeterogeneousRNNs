
"""
Find LIF reset voltages matching the QIF f-I curve or gain.
For different sequence parameters, the input current range is quite different.
Therefore, we find the best match per sequence target to keep the LIF and QIF
f-I curves as close as possible.

The function also saves an f-I curve plot for reference.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lif_exact_freq(I, vrest, vt, tau_m=1.0):
    """Compute the exact analytical firing rate (Hz) for the LIF model (dx/dt = -x + I).

    Parameters:
    -----------
    I : float or np.ndarray
        Injected current.
    vrest : float
        Resting / reset voltage.
    vt : float
        Threshold voltage.
    tau_m : float
        Membrane time constant (ms).
    """
    I = np.asarray(I, dtype=float)
    freqs = np.zeros_like(I)
    mask = I > vt

    if np.any(mask):
        I_v = I[mask]
        num = I_v - vrest
        den = I_v - vt
        t_isi = tau_m * np.log(num / den)  # ISI in ms
        freqs[mask] = 1000.0 / t_isi  # Firing rate in Hz

    return freqs


def qif_exact_freq(I, tau_m=1.0):
    """Compute the exact analytical firing rate (Hz) for the QIF / Theta model.

    Parameters:
    -----------
    I : float or np.ndarray
        Injected current.
    tau_m : float
        Membrane time constant (ms).
    """
    I = np.asarray(I, dtype=float)
    freqs = np.zeros_like(I)
    mask = I > 0.0

    if np.any(mask):
        freqs[mask] = (1000.0 * np.sqrt(I[mask])) / (np.pi * tau_m)

    return freqs


def simulate_single_cell(
    current,
    neuron_type,
    vreset=0.0,
    vt=0.0,
    dt=0.1,
    n_steps=1000,
    sigman=0.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Same initial-condition distributions as the network.
    if neuron_type == "QIF":
        x = rng.uniform(0, 2 * np.pi)
    elif neuron_type == "LIF":
        x = rng.uniform(vreset, vt)
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")

    spikes = 0

    for _ in range(n_steps):
        # Same noise convention as simulation.py:
        # one sample per timestep, shared across both RK2 stages.
        Itot = current + sigman * rng.standard_normal()

        if neuron_type == "LIF":
            # RK2 midpoint integration
            dx = -x + Itot
            xmid = x + dt * dx / 2

            dx = -xmid + Itot
            xnew = x + dt * dx

            # Same threshold crossing and reset
            if x < vt and xnew > vt:
                spikes += 1
                xnew = vreset

        else:  # QIF
            # RK2 midpoint integration
            dx = (
                1 - np.cos(x)
                + Itot * (1 + np.cos(x))
            )
            xmid = x + dt * dx / 2

            dx = (
                1 - np.cos(xmid)
                + Itot * (1 + np.cos(xmid))
            )
            xnew = x + dt * dx

            # Same phase-pi crossing detection
            dpi = np.mod(
                np.pi - np.mod(x, 2 * np.pi),
                2 * np.pi,
            )
            delta_x = xnew - x

            if (delta_x > 0) and (delta_x - dpi > 0):
                spikes += 1

            # No reset for QIF neurons

        x = xnew

    return spikes


def fi_curve(currents, neuron_type, vreset=0.0, vt=0.0,
             dt=0.1, n_steps=1000, sigman=0.0):
    simulation_seconds = n_steps * dt / 1000

    return np.array([
        simulate_single_cell(
            current, neuron_type, vreset=vreset, vt=vt,
            dt=dt, n_steps=n_steps, sigman=sigman, seed=i
        ) / simulation_seconds
        for i, current in enumerate(currents)
    ])


def fit_fi_gain(currents, rates, fit_min=0, fit_max=None, min_fit_points=5):
    """Fit a linear gain over the specified current interval."""
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
    fit_current_min=0.0,
    fit_current_max=None,
    current_scale=1.0,
    current_shift=0.0,
    current_min_floor=0.0,
    theorical=True,
):
    """Find optimal LIF reset voltages using theoretical or simulated f-I curves."""

    if vreset_values is None:
        vreset_values = np.linspace(-35, -5, 61)

    # Defining current range

    raw_min = current_scale * offset + current_shift
    raw_max = current_scale * (offset + amplitude) + current_shift

    if raw_max <= raw_min:
        raise ValueError("Amplitude must define a positive current interval.")

    shift = max(0.0, current_min_floor - raw_min)
    current_min = raw_min + shift
    current_max = raw_max + shift

    fit_min = max(current_min, fit_current_min)
    if fit_current_max is not None:
        current_max = min(current_max, fit_current_max)
    fit_max = current_max

    if fit_min >= fit_max:
        raise ValueError("No valid current interval for fitting the gain.")

    currents = np.linspace(current_min, current_max, n_currents)

    simulation_args = {
        "vt": vt,
        "dt": dt,
        "n_steps": n_steps,
        "sigman": sigman,
    }

    qif_simulation_rates = fi_curve(currents, "QIF", **simulation_args)
    qif_theory_rates = qif_exact_freq(currents)
    qif_rates = qif_theory_rates if theorical else qif_simulation_rates

    qif_fit = fit_fi_gain(
        currents, qif_rates, fit_min=fit_min, fit_max=fit_max
    )

    candidates = []

    for vreset in vreset_values:
        lif_simulation_rates = fi_curve(
            currents, "LIF", vreset=float(vreset), **simulation_args
        )
        lif_theory_rates = lif_exact_freq(
            currents,
            vrest=float(vreset),
            vt=vt,
        )
        lif_rates = lif_theory_rates if theorical else lif_simulation_rates

        lif_fit = fit_fi_gain(
            currents, lif_rates, fit_min=fit_min, fit_max=fit_max
        )

        errors = {
            name: rule(lif_rates, qif_rates, lif_fit, qif_fit)
            for name, rule in comparison_rules.items()
        }
        errors["overlap_simulation"] = overlap_error(
            lif_simulation_rates, qif_simulation_rates, None, None
        )
        errors["overlap_theorical"] = overlap_error(
            lif_theory_rates, qif_theory_rates, None, None
        )

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
        "theorical": bool(theorical),
    }

    best_simulation_overlap = min(
        candidates,
        key=lambda item: item["errors"]["overlap_simulation"],
    )
    best_theory_overlap = min(
        candidates,
        key=lambda item: item["errors"]["overlap_theorical"],
    )
    result["best_vreset_overlap_simulation"] = best_simulation_overlap["vreset"]
    result["error_overlap_simulation"] = best_simulation_overlap["errors"]["overlap_simulation"]
    result["best_vreset_overlap_theorical"] = best_theory_overlap["vreset"]
    result["error_overlap_theorical"] = best_theory_overlap["errors"]["overlap_theorical"]

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

    theory_currents = np.linspace(
        fi_result["current_min"],
        fi_result["current_max"],
        max(500, len(currents)),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    qif_rates = fi_curve(currents, "QIF", **simulation_args)
    ax.plot(
        currents, qif_rates, color="black", linewidth=2, label="QIF (simulation)"
    )
    ax.plot(
        theory_currents,
        qif_exact_freq(theory_currents),
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="QIF (theory)",
    )

    simulation_vreset = fi_result["best_vreset_overlap_simulation"]
    simulation_lif_rates = fi_curve(
        currents, "LIF", vreset=simulation_vreset, **simulation_args
    )
    ax.plot(
        currents,
        simulation_lif_rates,
        color="tab:blue",
        linewidth=1.8,
        label=f"LIF (simulation overlap, Vreset={simulation_vreset:.2f})",
    )

    theory_vreset = fi_result["best_vreset_overlap_theorical"]
    theory_lif_rates = lif_exact_freq(
        theory_currents,
        vrest=theory_vreset,
        vt=fi_settings["vt"],
    )
    ax.plot(
        theory_currents,
        theory_lif_rates,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label=f"LIF (theory overlap, Vreset={theory_vreset:.2f})",
    )

    for label, config in zip(labels, configs):
        if label == "overlap":
            continue
        lif_rates = fi_curve(
            currents, "LIF", vreset=config["vreset"],
            **{**simulation_args, "vt": config["vt"]},
        )
        lif_line, = ax.plot(
            currents, lif_rates, linewidth=1.8,
            label=f"LIF ({label}, Vreset={config['vreset']:.2f}, simulation)",
        )
        ax.plot(
            theory_currents,
            lif_exact_freq(
                theory_currents,
                vrest=config["vreset"],
                vt=config["vt"],
            ),
            color=lif_line.get_color(),
            linestyle="--",
            linewidth=1.8,
            label=f"LIF ({label}, Vreset={config['vreset']:.2f}, theory)",
        )

    overlap_summary = (
        f"Simulation MSE={fi_result['error_overlap_simulation']:.4g} "
        f"(Vr={simulation_vreset:.2f})\n"
        f"Theory MSE={fi_result['error_overlap_theorical']:.4g} "
        f"(Vr={theory_vreset:.2f})"
    )
    ax.text(
        0.02,
        0.98,
        overlap_summary,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
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

