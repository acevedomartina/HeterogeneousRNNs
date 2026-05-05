# ------------------------------------------------------------
# Connectivity matrices
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
import random
import os

# ------------------------------------------------------------
# Style

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

# IDUN
# base_path = Path(r"J:\new_target")


# Home
# (r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")

# Original
base_path = SCRIPT_DIR 

base_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------


########## Loop variables ##########
vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
namelist_slope = [f"Gain = {f}" for f in slope]

slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]

pqif_number = [0, 0.25, 0.5, 0.75, 1]
namelist_pqif = [f"pqif={p}" for p in pqif_number]

pqif_homogenous = [0, 1]
dynamics = ['oscillations', 'sequences']
dynamics = ['oscillations']
# seed_number = [i for i in range(0, 50)]
seed = random.randint(1, 49)
print(f"Used seed {seed}")

# ------------------------------------------------------------

cmap = plt.cm.coolwarm.copy()
cmap.set_bad(color="white")

row_labels = []


for dyn in dynamics:

        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        figname = f"{dyn}_connectivity_matrix.svg"

        simulation_number = ([i for i in range(1,5)]) if dyn == 'oscillations' else ([i for i in range(9,13)])

        for idx, (sim, vr, f) in enumerate(zip(simulation_number, vrest, slope)):

                for jdx, pqif in enumerate(pqif_number):
                        ax = axes[idx, jdx]

                        simulation_path = f"{base_path}\\{dyn}\\simulation_{sim}\\simulation_{sim}_connectivity_matrix"
                        # simulation_path = f"{base_path}\\simulation_{sim}\\simulation_{sim}_connectivity_matrix"

                        J = np.loadtxt(f"{simulation_path}\\simulation_{sim}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}", delimiter=",")
                        
                        i, j = np.nonzero(J)  # return array of indices where matrix is not zero

                        w = J[i, j]

                        # s = 20*np.abs(w)

                        s = 2*np.abs(w)

                        sc = ax.scatter(j, i, c=w, cmap="coolwarm", s=s, vmin=-0.5, vmax=0.5)  # x = presynaptic index = j, y = postsynaptic index = i
                        # sc = ax.scatter(j, i, c=w, cmap="coolwarm", s=10, vmin=-0.5, vmax=0.5)  # x = presynaptic index = j, y = postsynaptic index = i
                        
                        
                        # ax.set_title(f"pqif={pqif}, Gain = {f}")

                        ax.set_xlabel("presynaptic (j)")
                        ax.set_ylabel("post")
                        ax.invert_yaxis()
                        ax.set_aspect("equal")

                        if jdx == 0:
                                # ax.set_ylabel(f"{namelist_slope[idx]}\npostsynaptic (i)", fontsize=14, fontweight='bold')
                                ax.set_ylabel(
                                        f"$\\mathbf{{{namelist_slope[idx]}}}$\npostsynaptic (i)",
                                        fontsize=14
                                        )
                        if idx == 0:
                                ax.set_title(namelist_pqif[jdx], fontsize=14, fontweight='bold')


                        if idx < 3:
                                ax.set_xlabel("")

                        if jdx > 0:
                                ax.set_ylabel("")



        plt.suptitle(f"Connectivity matrices when target is {dyn}")
        plt.colorbar(sc, ax=axes.ravel().tolist(), label="Weight")
        # plt.tight_layout()
        fig_path = FIGURES_DIR / f"{dyn}_connectivity_matrix.svg"
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved in '{fig_path}'")
        # plt.show()


