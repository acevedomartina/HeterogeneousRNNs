# ------------------------------------------------------------
# Spike train for all pqif
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
import os

# ------------------------------------------------------------

# Path

# Python script
import pathlib, os
plt.style.use(os.path.join(pathlib.Path(__file__).parent, 'plos.mplstyle'))

# Jupyter script
# style_path = Path.cwd() / "plos.mplstyle"   # cwd = folder Jupyter launched from
# plt.style.use(str(style_path))


# ------------------------------------------------------------

# Formatting

def make_color_map(values, cmap, vmin=0.25, vmax=1.0):
    '''
    Function that takes array of values and make a colormap for them
    '''
    values_sorted = sorted(values)
    colors = cmap(np.linspace(vmin, vmax, len(values)))
    return dict(zip(values_sorted, colors))

# To be given value
v_rest = [-22, -17, -12.3, -8.5]
pqif_vector = [0, 0.25, 0.5, 0.75, 1]
slope = [14.44, 10.68, 8.65, 7.18]

# Make colors once
color_map_vrest = make_color_map(v_rest, plt.cm.Reds)  # Reds for vrest
color_map_slope = make_color_map(slope, plt.cm.Reds)  # Reds for vrest
color_map_pqif = make_color_map(pqif_vector[1:-1], plt.cm.Greens)  # Greens for pqif

# Explicit endpoints for pqif
color_map_pqif[pqif_vector[0]] = 'black'
color_map_pqif[pqif_vector[-1]] = 'steelblue'

# ------------------------------------------------------------

# Path

# --- File path handling
SCRIPT_DIR = Path(__file__).resolve().parent  # .py
# SCRIPT_DIR = Path.cwd()  # .ipynb
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)   # creates ./figures/ if needed

# --- Base path
# base_path = Path(r"J:\new_target")  # IDUN
# base_path = Path(r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")  # home
base_path = SCRIPT_DIR  # original
base_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------

# Loop variables

quadrant_names = ["Q1", "Q2", "Q3", "Q4"]
vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
simulation_number = [i for i in range(1,5)]
slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]
pqif_number = [0, 0.25, 0.5, 0.75, 1]
# pqif_number = [0.25, 0.5, 0.75]

# dynamics = ["oscillations"]
dynamics = ["oscillations", "sequences"]

seeds = range(2)
N_neurons = 200 # number of neurons

# ------------------------------------------------------------

# Functions

def load_matrix(path):
    return np.genfromtxt(path, delimiter=',')

def slice_neurons(output, pqif):
    '''
    Slice output based on pqif. First pqif*N are QIF and remaining LIF.
    '''

    N = output.shape[1]  # Number of columns --> Number of neurons


    h = int(round(pqif * N))

    # All rows (time), QIF and LIF
    QIF = output[:, :h]  # shape (T, h) - first h columns
    LIF = output[:, h:]  # shape (T, N-h) - remaining columns


    return QIF, LIF, N


def fill_zero(QIF, LIF, pqif, N_neurons):
    ''' Function that fills the output arrays with zeros where the neuron is not'''

    pqif_indices = [0 for i in range(0,N_neurons)]  # Array with zeros for all neuron indices

    qif_indices = int(pqif * N_neurons)
    # print(f"Cutoff: {qif_indices}")


    qif_filled = pqif_indices.copy()
    qif_filled[:qif_indices] = QIF  # first qif_indices are QIF

    lif_filled = pqif_indices.copy()
    lif_filled[qif_indices:] = LIF  # lif is from qif_indices and out

    return qif_filled, lif_filled


# ------------------------------------------------------------

# Plot output across different pqif


for dyn in dynamics:  # If not both dynamics, change above in parameters

    simulation_number = [i for i in range(1,5)] if dyn == "oscillations" else [i for i in range(9,13)]


    fig, axs = plt.subplots(
    nrows=1,
    ncols=len(pqif_number),
    figsize=(20, 8),          
    constrained_layout=True   
    )

    for idx_pqif, pqif in enumerate(pqif_number):

        ax = axs[idx_pqif]

        for idx, f in zip(simulation_number, slope):

            label = f


            simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
            # simulation_path = f"{base_path}\\{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


            # simulation_path = f"{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


            ###### OUTPUT FILES

            simulation_path_output = f"{base_path}\\simulation_{idx}\\simulation_{idx}_outputs"
            # simulation_path_output = f"{base_path}\\{dyn}\\simulation_{sim}\\simulation_{sim}_outputs"




            qif_mean_output = []
 

            lif_mean_output = []



            for seed in seeds:  # Loop over seeds

                # Paths

                
                path_output = f"{simulation_path_output}\\simulation_{idx}_outputs_pqif_{pqif}_iloop_11_seed_{seed}.csv"

                # Load output matrix
                output = load_matrix(path_output)

                # Assign portions of output matrix to QIF and LIF variables
                QIF, LIF, N_neurons = slice_neurons(output, pqif)

                # average over time (so one element is that neurons time averaged output)
                QIF_time_means = np.nanmean(QIF, axis=0)
                LIF_time_means = np.nanmean(LIF, axis=0)

                # to list

                qif_array = QIF_time_means.tolist()
                lif_array = LIF_time_means.tolist()

                # print(f"Averaged over time:\nQIF: {qif_array}\nLIF{lif_array}:")


                # Fill zeros
                qif_filled, lif_filled = fill_zero(qif_array, lif_array, pqif, N_neurons)

                # print(f"QIF: {qif_filled}")
                # print(F"LIF: {lif_filled}")

                # For each seed, append to outer list
                qif_mean_output.append(qif_filled)
                lif_mean_output.append(lif_filled)

            # Make numpy array across seeds
            qif_arr = np.array(qif_mean_output)      # shape (n_seeds, N_neurons)
            lif_arr = np.array(lif_mean_output)      # shape (n_seeds, N_neurons)


            # Mean +- std across seeds
            qif_mean   = np.mean(qif_arr, axis=0)               # (N_neurons,)
            qif_std    = np.std(qif_arr, axis=0, ddof=1)

            lif_mean   = np.mean(lif_arr, axis=0)               # (N_neurons,)
            lif_std    = np.std(lif_arr, axis=0, ddof=1)

            
            # ------------------------------------------------------------
            # Plotting

            col = color_map_slope[f]
            if pqif == 1:
                col = 'steelblue'

            # ---- QIF ----------------------------------------------------
            if pqif > 0:          # there are QIF neurons for this pqif
                ax.plot(
                    np.arange(N_neurons),      # x‑axis = neuron index
                    qif_mean,
                    # yerr=qif_std,
                    # fmt='o',
                    color='steelblue',
                    # markersize=4,
                    # capsize=3,
                    label='QIF' if (idx == simulation_number[0] and pqif == 1) else None,
                    # elinewidth=1.2,
                )

            # ---- LIF ----------------------------------------------------
            if pqif < 1:          # there are LIF neurons for this pqif
                ax.plot(
                    np.arange(N_neurons),
                    lif_mean,
                    # yerr=lif_std,
                    # fmt='s',
                    color=col,
                    # markersize=4,
                    # capsize=3,
                    label=f'{f}' if (pqif not in (0, 1) and idx == simulation_number[0]) else None,
                    # elinewidth=1.2,
                )


            ax.set_box_aspect(1)
    fig_path = FIGURES_DIR / "output" / f"{dyn}_output_all.svg"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved in '{fig_path}'")
    plt.suptitle(f"{dyn.capitalize()}: Mean over time of output of presynaptic neurons", fontsize=20, weight='bold')
    # plt.legend()


    plt.show()



