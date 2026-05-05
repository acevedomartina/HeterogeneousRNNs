# ------------------------------------------------------------
# Connectivity in subpopulations
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

## Reusable color utility

import numpy as np
import matplotlib.pyplot as plt

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

# File path handling
SCRIPT_DIR = Path(__file__).resolve().parent
# NPZ_PATH = SCRIPT_DIR / "fi_data.npz"  
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)   # creates ./figures/ if needed


# Choose path
# base_path = Path(r"J:\new_target")
base_path = Path(r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")
base_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------

quadrant_names = ["Q1", "Q2", "Q3", "Q4"]


vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
simulation_number = [i for i in range(9,13)]
slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]
pqif_number = [0, 0.25, 0.5, 0.75, 1]

quantifications = ["mean", "std"]

seeds = range(10)

def load_matrix(path):
    return np.genfromtxt(path, delimiter=',')

def get_quadrants(W):
    N = W.shape[0]
    h = N // 2
    return [
        W[:h, :h],
        W[:h, h:],
        W[h:, :h],
        W[h:, h:]
    ]



for q in quantifications:
    for pqif in pqif_number:

        # New figure per pqif iteration
        fig, ax = plt.subplots(figsize=(6,4))

        for idx, f in zip(simulation_number, slope):

            label = f
            simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
            # if idx == 1: 
            #     label = 14.44
            # if idx == 2: 
            #     label = 7.18


            means_per_seed = []
            stds_per_seed = [] 

            global_means_per_seed = []
            global_stds_per_seed = []

            for seed in seeds:
                # paths
                path0 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_0_seed_{seed}"
                path11 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}"
                
                # Load the initialized matrix and the matrix after training
                W0 = load_matrix(path0)
                W11 = load_matrix(path11)
                
                # Mask zeros
                mask_zero = np.isclose(W0, 0.0)
                W11_masked = np.where(mask_zero, np.nan, W11)
                
                # Get quadrants from masked matrix
                quadrants = get_quadrants(W11_masked)
                
                # Containers for mean and standard deviation per quadrant
                means = []
                stds = []
                
                for Q in quadrants:
                    # Take mean and standard deviation of each quadrant, append to list
                    # mean and stds will have 4 values, one per quadrant
                    data = Q[~np.isnan(Q)]
                    means.append(np.mean(data))
                    stds.append(np.std(data))

        
                # Append to outer list containing the mean and stds for all seeds
                means_per_seed.append(means)
                stds_per_seed.append(stds)

                # Global mean and standard deviation
                global_data = W11_masked[~np.isnan(W11_masked)]   # flatten, drop NaNs
                global_means_per_seed.append(np.mean(global_data))
                global_stds_per_seed.append(np.std(global_data))

            # Convert to numpy array
            means_per_seed = np.array(means_per_seed)
            stds_per_seed = np.array(stds_per_seed)

            # Get mean and standard deviation across seeds for both quantifications
            # MEAN:
            mean_of_means = np.mean(means_per_seed, axis=0)
            std_of_means = np.std(means_per_seed, axis=0)

            # STD:
            mean_of_stds = np.mean(stds_per_seed, axis=0)
            std_of_stds = np.std(stds_per_seed, axis=0)


            # Global mean
            global_mean_mean = np.mean(global_means_per_seed)
            global_std_mean = np.std(global_means_per_seed)

            # Global std
            global_mean_std = np.mean(global_stds_per_seed)
            global_std_std = np.std(global_stds_per_seed)
            # Plot for both


            if q == "mean":
                y = mean_of_means
                yerror = std_of_means
                ylim = (-0.5, 0.5)

                global_y = global_mean_mean
                global_yerr = global_std_mean

            elif q == "std":
                y = mean_of_stds
                yerror = std_of_stds
                ylim = (0.25, 0.65)

                global_y = global_mean_std
                global_yerr = global_std_std

            print(f"Plotting for {pqif}")

            
            ax.errorbar(
                quadrant_names,
                y,
                yerr=yerror,
                marker='o',
                capsize=5,
                label=f'Gain = {label}',
                color=color_map_slope[f]
            )

            # shaded area for global
            ax.axhspan(global_y - global_yerr, global_y + global_yerr, alpha=0.5, color=color_map_slope[f])



        ax.set_xlabel("Quadrant")
        ax.set_ylabel(q)
        ax.set_title(f"Sequences, pqif: {pqif}")
        # plt.legend()
        ax.set_ylim(ylim)
        # plt.tight_layout()
        fig_path = FIGURES_DIR / f"sequences_{pqif}_subpopulations_{q}.svg"
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved in '{fig_path}'")
        # plt.show()
        