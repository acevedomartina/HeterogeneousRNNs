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

# ------------------------------------------------------------

quadrant_names = ["Q1", "Q2", "Q3", "Q4"]


vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
simulation_number = [i for i in range(1,5)]
slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]
pqif_number = [0, 0.25, 0.5, 0.75, 1]


base_path = Path(r"J:\new_target")

seeds = range(50)

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



# for idx in range(1,3):
for pqif in pqif_number:
    for idx, f in zip(simulation_number, slope):
        label = f
        simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
        # if idx == 1: 
        #     label = 14.44
        # if idx == 2: 
        #     label = 7.18

        means_per_seed = []
        stds_per_seed = [] 
        for seed in seeds:
            # paths
            path0 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_0_seed_{seed}"
            path11 = f"{simulation_path}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}"
            
            # cargar
            W0 = load_matrix(path0)
            W11 = load_matrix(path11)
            
            # máscara de ceros
            mask_zero = np.isclose(W0, 0.0)
            W11_masked = np.where(mask_zero, np.nan, W11)
            
            # cuadrantes
            quadrants = get_quadrants(W11_masked)
            
            means = []
            stds = []
            
            for Q in quadrants:
                data = Q[~np.isnan(Q)]
                means.append(np.mean(data))
                stds.append(np.std(data))


    
            means_per_seed.append(means)
            stds_per_seed.append(stds)

        # promedio sobre seeds
        means_per_seed = np.array(means_per_seed)
        stds_per_seed = np.array(stds_per_seed)

        mean_of_means = np.mean(means_per_seed, axis=0)
        std_of_means = np.std(means_per_seed, axis=0)

        mean_of_stds = np.mean(stds_per_seed, axis=0)
        std_of_stds = np.std(stds_per_seed, axis=0)

        
        plt.errorbar(
            quadrant_names,
            mean_of_stds,
            yerr=std_of_stds,
            marker='o',
            capsize=5,
            label=f'Gain = {label}',
            color=color_map_slope[f]
        )

    plt.xlabel("Quadrant")
    plt.ylabel(f"$\sigma$W")
    plt.title(f"pqif: {pqif}")
    plt.legend()
    plt.ylim(0.25, 0.65)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"oscillations_{pqif}_subpopulations.svg"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved in '{fig_path}'")
    plt.show()
    plt.show()