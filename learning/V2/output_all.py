# ------------------------------------------------------------
# Spike train for all pqif
# This code is not finished
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
# plt.style.use(os.path.join(pathlib.Path(__file__).parent, 'plos.mplstyle'))

# Jupyter script
style_path = Path.cwd() / "plos.mplstyle"   # cwd = folder Jupyter launched from
plt.style.use(str(style_path))


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
# SCRIPT_DIR = Path(__file__).resolve().parent  # .py
SCRIPT_DIR = Path.cwd()  # .ipynb
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)   # creates ./figures/ if needed

# --- Base path
# base_path = Path(r"J:\new_target")  # IDUN
base_path = Path(r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")  # home
# base_path = SCRIPT_DIR  # original
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
pqif_number = [0.25, 0.5, 0.75]

# quantifications = ["mean", "std"]
quantifications = ["mean"]
# dynamics = ["sequences"]  
# dynamics = ["oscillations"]
dynamics = ["oscillations", "sequences"]

seeds = range(10)
N = 200 # number of neurons

# ------------------------------------------------------------

# Functions

def load_matrix(path):
    return np.genfromtxt(path, delimiter=',')

def get_quadrants(W, pqif):
    ''' 
    Split into quadrants based on pqif. If homogenous, split into equal sizes (we can adjust this if we want different sizes later) 

    Parameters
    --------------
    W :  np.ndarray
        Connectivity matrix
    pqif : float
        fraction of QIF neurons, determines where to split the matrices
    
    Returns
    --------------
    list of np.ndarray
        [Q1, Q2, Q3, Q4] where
            Q1 = top‑left,
            Q2 = top‑right,
            Q3 = bottom‑left,
            Q4 = bottom‑right.
    '''
    N = W.shape[0]

    # Decide where to cut matrix
    # h = N // 2  # If we want all to be equal sizes
    if 0 < pqif < 1:
        h = int(round(pqif * N))
    else:
        h = N // 2

    return [
        W[:h, :h],  # top-left
        W[:h, h:],  # top-right
        W[h:, :h],  # bottom-left
        W[h:, h:]  # bottom-right
    ]

def slice_neurons(output, pqif):
    '''
    Slice output based on pqif. First pqif*N are QIF and remaining LIF.
    '''

    N = output.shape[1]  # Number of columns --> Number of neurons


    h = int(round(pqif * N))

    # All rows (time), QIF and LIF
    QIF = output[:, :h]  # shape (T, h) - first h columns
    LIF = output[:, h:]  # shape (T, N-h) - remaining columns


    return QIF, LIF

# def pre_post(QIF, LIF):
#     """
#     Return a list that tells which output matrix (T, pre) belongs to each quadrant.
#     """
#     return [QIF, LIF, QIF, LIF]       # Q1,Q2,Q3,Q4 in that order


for dyn in dynamics:  # If not both dynamics, change above in parameters
    simulation_number = [i for i in range(1,5)] if dyn == "oscillations" else [i for i in range(9,13)]
    for q in quantifications:

        fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(20, 8),          
        constrained_layout=True   
        )

        for idx_pqif, pqif in enumerate(pqif_number):

            ax = axs[idx_pqif]

            for idx, f in zip(simulation_number, slope):

                label = f


                ###### CONNECTIVITY MATRIX

                simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
                # simulation_path = f"{base_path}\\{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


                # simulation_path = f"{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


                ###### OUTPUT FILES

                simulation_path_output = f"{base_path}\\simulation_{idx}\\simulation_{idx}_outputs"
                # simulation_path_output = f"{base_path}\\{dyn}\\simulation_{sim}\\simulation_{sim}_outputs"



                means_per_seed = []
                stds_per_seed = [] 

                global_means_per_seed = []
                global_stds_per_seed = []

                QIF_means_per_seed = []
                QIF_stds_per_seed = []
                LIF_means_per_seed = []
                LIF_stds_per_seed = []

                # Containers for mean and standard deviation for outputs
                means_presyn = []
                stds_presyn = []


                for seed in seeds:
                    # Paths

                    path0 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_0_seed_{seed}"
                    path11 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}"
                    
                    path_output = f"{simulation_path_output}\\simulation_{idx}_outputs_pqif_{pqif}_iloop_11_seed_{seed}.csv"



                    # Load the initialized matrix and the matrix after training
                    W0 = load_matrix(path0)
                    W11 = load_matrix(path11)
                    
                    # Mask zeros
                    mask_zero = np.isclose(W0, 0.0)
                    W11_masked = np.where(mask_zero, np.nan, W11)
                    
                    # Get quadrants from masked matrix
                    quadrants = get_quadrants(W11_masked, pqif)
                    
                    # Load output matrix
                    output = load_matrix(path_output)

                    # Assign portions of output matrix to QIF and LIF variables
                    QIF, LIF = slice_neurons(output, pqif)

                    # Assign QIF and LIF as presynaptic neurons to correct quadrants
                    # presynaptic_neurons = pre_post(QIF, LIF)  # Becomes a list 

                    # average over time (so one element is that neurons time averaged output)
                    QIF_time_means = np.nanmean(QIF, axis=0)
                    LIF_time_means = np.nanmean(LIF, axis=0)

                    # average over the neurons belonging to this quadrant → scalar
                    # QIF_means = np.nanmean(QIF_time_means)
                    # LIF_means = np.nanmean(LIF_time_means)

                    # Append to seeds
                    QIF_means_per_seed.append(QIF_time_means)
                    LIF_means_per_seed.append(LIF_time_means)

                

                # ------------------------------------------------------------

                # Collapse presynaptic activity across seeds

                QIF_means_per_seed = np.array(QIF_means_per_seed) # convert to numpy
                LIF_means_per_seed = np.array(LIF_means_per_seed)  # convert to numpy

                QIF_means_per_seed = np.mean(QIF_means_per_seed, axis=0)  # means for qif
                QIF_stds_per_seed = np.std(QIF_stds_per_seed, axis=0, ddof=1)  # std to get errorbars

                LIF_means_per_seed = np.mean(QIF_means_per_seed, axis=0)  # means for qif
                LIF_stds_per_seed = np.std(QIF_stds_per_seed, axis=0, ddof=1)  # std to get errorbars


                # Conditions if not pqif==1
                color=(color_map_slope[f] if pqif != 1 else 'steelblue')
                y = LIF_means_per_seed
                yerror = LIF_stds_per_seed
                label = f"{f}"

                # Determine x axis 
                if 0 > pqif < 1:
                    x = pqif * N
                else:
                    x = N
                
                # Plot
                if pqif == 1:
                    color = 'steelblue'
                    y = QIF_means_per_seed
                    yerr = QIF_stds_per_seed
                    label = "QIF"
                    

                    if idx != simulation_number[0]:  
                        # only take the first instance of QIF, since homogenous QIF does not change per simulation outside of variance
                        # also to not have more seeds for QIF than the other cases
                        continue 



                ax.errorbar(x, y, yerr=yerror, capsize=5, color=color, elinewidth=1.2, markersize=6, label=label)
                    

            ax.set_xlabel("Neurons 1,...,N")
            ax.set_ylabel("Presynaptic activity (mean +- std)")
            ax.set_title(f"pqif: {pqif}")
            ax.set_box_aspect(1)
            # yticks = np.arange(start=0.18, stop=0.62, step=0.1)
            # ax.set_ylim(0.18, 0.62)
            ax.set_xticks(np.arange(N))
            # ax.set_yticks(yticks)
        fig_path = FIGURES_DIR / f"{dyn}_output_subpopulations.svg"
        # plt.savefig(fig_path, dpi=300)
        # print(f"Figure saved in '{fig_path}'")
        # plt.suplots_adjust(hspace=None)
        plt.suptitle(f"{dyn.capitalize()}: Mean +- $\sigma$ of output of presynaptic neuron per quadrant", fontsize=20, weight='bold')
        plt.legend()


plt.show()

