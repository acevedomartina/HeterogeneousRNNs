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
FIGURES_DIR = SCRIPT_DIR / "final_figures_thesis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)   # creates ./figures/ if needed


# Choose path

# IDUN
# base_path = Path(r"J:\new_target")

base_path = Path(r"J:\ordinary_simulations")  # ordinary simulations
# base_path = Path(r"J:\target_versions_oscillations\target_4")  # ordinary simulations
# J:\target_versions_oscillations\target_1


# Home
# (r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")

# Original
# base_path = SCRIPT_DIR 

base_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------

quadrant_names = ["Q1", "Q2", "Q3", "Q4"]


vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
simulation_number = [i for i in range(1,5)]
slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]
pqif_number = [0, 0.25, 0.5, 0.75, 1]

# quantifications = ["std"]
# dynamics = ["sequences"]  
quantifications = ["mean", "std"]
# dynamics = ["oscillations"]
dynamics = ["oscillations", "sequences"]

seeds = range(50)

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


for dyn in dynamics:  # If not both dynamics, change above in parameters
    simulation_number = [i for i in range(1,5)] if dyn == "oscillations" else [i for i in range(9,13)]
    for q in quantifications:
        for pqif in pqif_number:

            # New figure per pqif iteration
            fig, ax = plt.subplots(figsize=(6,4))

            for idx, f in zip(simulation_number, slope):

                label = f
                simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
                # simulation_path = f"{base_path}\\{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


                # simulation_path = f"{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


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
                    quadrants = get_quadrants(W11_masked, pqif)
                    
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
                    ax.set_yticks(np.linspace(0, 0.26, 3))
                    ylim = (-0.05, 0.3)

                    global_y = global_mean_mean
                    global_yerr = global_std_mean

                elif q == "std":
                    y = mean_of_stds
                    yerror = std_of_stds
                    ylim = (0.25, 0.75)
                    ax.set_yticks(np.linspace(0.35, 0.65, 3))

                    global_y = global_mean_std
                    global_yerr = global_std_std

                print(f"Plotting for sim {idx}, pqif {pqif} from {base_path}")

                if pqif == 1:
                    # color = 'steelblue'

                    if idx != simulation_number[0]:  
                        # only take the first instance of QIF, since homogenous QIF does not change per simulation outside of variance
                        # also to not have more seeds for QIF than the other cases
                        continue # only

                
                ax.errorbar(
                    quadrant_names,
                    y,
                    yerr=yerror,
                    marker='o',
                    capsize=5,
                    label=f'Gain = {label}',
                    color=(color_map_slope[f] if pqif != 1 else 'steelblue')
                )

                # shaded area for global
                ax.axhspan(global_y - global_yerr, global_y + global_yerr, alpha=0.4, color=(color_map_slope[f] if pqif != 1 else 'steelblue'))



            ax.set_xlabel("Quadrant")
            ax.set_ylabel(q)
            ax.set_title(f"{dyn.capitalize()} shared target, pqif: {pqif}")
            # plt.legend()
            ax.set_ylim(ylim)
            ax.set_box_aspect(1)
            # plt.tight_layout()
            fig_path = FIGURES_DIR / f"{dyn}_{pqif}_subpopulations_{q}.svg"
            plt.savefig(fig_path, dpi=300)
            print(f"Figure saved in '{fig_path}'")
            # plt.show()

            # oscillations\simulation_1\simulation_1_connectivity_matrix\simulation_1_connectivity_pqif_0_iloop_0_seed_0
        
            # oscillations\simulation_1\simulation_1_connectivity_matrix\simulation_1_connectivity_pqif_0_iloop_0_seed_0