# ------------------------------------------------------------
# Fit f-I curves
# ------------------------------------------------------------

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib
from brokenaxes import brokenaxes
import matplotlib.lines as mlines
from pathlib import Path

# Load matplotlib style
# plt.style.use('plos.mplstyle')

import pathlib, os
plt.style.use(os.path.join(pathlib.Path(__file__).parent, 'plos.mplstyle'))

# ------------------------------------------------------------

# Simulation parameters
T = 1000
dt = 0.1
sigman = 0.0
v_threshold = 0
v_rest = [-22, -12.3, -17, -8.5]

# ------------------------------------------------------------

# Formatting stuff

figsize=(20, 20)

# ------------------------------------------------------------

# File path handling
SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR / "fi_data.npz"  
FIGURES_DIR = SCRIPT_DIR / "final_figures_thesis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)   # creates ./figures/ if needed
# ------------------------------------------------------------

# Functions to simulate the dynamics of single neurons

def dynamics(x, I, tipo):
    if tipo == "LIF":
        dx = -x + I + np.random.randn() * sigman
    elif tipo == "QIF":
        dx = 1 - np.cos(x) + I * (1 + np.cos(x)) + np.random.randn() * sigman
    return dx

def detect(x_old, x_new, tipo, vt, vrest):
    spike = False
    if tipo == "LIF":
        if x_old < vt and x_new > vt:
            spike = True
            x_new = vrest
    elif tipo == "QIF":
        dpi = (np.pi - (x_old % (2*np.pi))) % (2*np.pi)
        if (x_new - x_old) > dpi:
            spike = True
    return x_new, spike

def simulate_single_cell(I, tipo, vt=None, vrest=0):
    x = vrest
    spikes = 0
    for _ in range(T):
        dx = dynamics(x, I, tipo)
        x_new = x + dt * dx
        x_new, spike = detect(x, x_new, tipo, vt, vrest)
        if spike:
            spikes += 1
        x = x_new
    return spikes / (T / 1000)  # frequency in Hz

# ------------------------------------------------------------

# Input currents
I_vals = np.linspace(0, 4, 20)
I_qif = np.linspace(0, 4, 20)

# ------------------------------------------------------------

# Store LIF firing rates
freqs_lif = {}
for vrest_value in v_rest:
    freqs = np.array([simulate_single_cell(I, "LIF", v_threshold, vrest=vrest_value) for I in I_vals])
    freqs_lif[vrest_value] = freqs

# Store QIF firing rates
freqs_qif = np.array([simulate_single_cell(I, "QIF") for I in I_qif])

# ------------------------------------------------------------

# Save everything to an npz file
np.savez(NPZ_PATH, I_vals=I_vals, v_rest=v_rest, freqs_lif=freqs_lif, I_qif=I_qif, freqs_qif=freqs_qif)
print(f"Data saved in {NPZ_PATH}")


# Load data
data = np.load(NPZ_PATH, allow_pickle=True)
I_vals = data['I_vals']
v_rest = data['v_rest']
freqs_lif = data['freqs_lif'].item()  # dict
I_qif = data['I_qif']
freqs_qif = data['freqs_qif']

# ------------------------------------------------------------

#Colors for plotting
v_rest = [-22, -17, -12.3, -8.5]
cmap = plt.cm.Reds
vrest_sorted = sorted(v_rest)
colors_sorted = [cmap(x) for x in np.linspace(0.25, 1, len(v_rest))]
color_map = {vr: c for vr, c in zip(vrest_sorted, colors_sorted)}

# vrest_figures = [-17, -12.3]
vrest_figures = [-22, -17, -12.3, -8.5]
# figsize = (3,3)



vrest_sorted = sorted(v_rest)

# fig, ax = plt.subplots(figsize=figsize)  # large figure
fig, ax = plt.subplots()  


# ------------------------------------------------------------

# Plot LIF
handles_dict = {}
labels_dict = {}
slopes_dict = {}

for vrest_value in vrest_sorted:
    freqs = freqs_lif[vrest_value]
    mask = I_vals > 0.7
    coef, cov = np.polyfit(I_vals[mask], freqs[mask], 1, cov=True)
    pendiente = coef[0]
    slopes_dict[vrest_value] = pendiente
    pendiente_err = np.sqrt(cov[0,0])


    if np.isin(vrest_value, vrest_figures):
        line, = ax.plot(I_vals, freqs, '-', color=color_map[vrest_value])
        ax.plot(I_vals[mask], np.polyval(coef, I_vals[mask]), ':', lw=3, color=color_map[vrest_value])

        handles_dict[vrest_value] = line
        labels_dict[vrest_value] = f"Vr={vrest_value}, {pendiente:.2f}±{pendiente_err:.2f}"

# Plot QIF
mask_qif = I_qif > 0.7
coef_qif, cov_qif = np.polyfit(I_qif[mask_qif], freqs_qif[mask_qif], 1, cov=True)
pendiente_qif = coef_qif[0]
slopes_dict['QIF'] = pendiente_qif
pendiente_qif_err = np.sqrt(cov_qif[0,0])

line_qif, = ax.plot(I_qif, freqs_qif, '-', color='steelblue')
ax.plot(I_qif[mask_qif], np.polyval(coef_qif, I_qif[mask_qif]), ':', lw=3, color='steelblue')

handles_dict['QIF'] = line_qif
labels_dict['QIF'] = f"QIF, {pendiente_qif:.2f}±{pendiente_qif_err:.2f}"

# Order handles and labels using vrest_figures (not vrest_sorted)
handles_colors = [handles_dict[vr] for vr in vrest_figures] + [handles_dict['QIF']]
labels_colors = [labels_dict[vr] for vr in vrest_figures] + [labels_dict['QIF']]

# Axis labels
ax.set_xlabel("Current (nA)")
ax.set_ylabel("Firing rate (Hz)")
ax.tick_params(axis='both', direction='in')

# FI and fit legend
line_solid = mlines.Line2D([0],[0], color='gray', lw=3)
line_dotted = mlines.Line2D([0],[0], color='gray', lw=3, linestyle=':')
leg1 = ax.legend(handles=[line_solid, line_dotted], labels=['FI curve', 'Fit'], loc='lower right', bbox_to_anchor=(1,0.37))

# Slopes legend - use vrest_figures instead of vrest_sorted
handles_dotted = [mlines.Line2D([], [], color=color_map[vr], linestyle=':', lw=3) for vr in vrest_figures] + [mlines.Line2D([], [], color='steelblue', linestyle=':', lw=3)]
leg2 = ax.legend(handles=handles_dotted, labels=labels_colors, loc='lower right', bbox_to_anchor=(1,0), title='Gain')

ax.add_artist(leg1)
ax.add_artist(leg2)

fig_path = FIGURES_DIR / "fi_curves_with_slopes.svg"
plt.savefig(fig_path, dpi=300)
print(f"Figure saved in '{fig_path}'")
plt.show()