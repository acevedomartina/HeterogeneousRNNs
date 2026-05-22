# ------------------------------------------------------------
# Spike train in subpopulations
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
# pqif_number = [0.25, 0.5, 0.75]

# quantifications = ["mean", "std"]
quantifications = ["mean"]
# dynamics = ["sequences"]  
# dynamics = ["oscillations"]
dynamics = ["oscillations", "sequences"]

seeds = range(10)

# choose a small distance
dx = 0.1 # half‑distance from the centre

# centre of the two points (kept at 1.0 so the x‑axis still shows “1” and “2”)
x_center = 1.0

# actual positions that will be used for the error‑bars
x_qif = x_center - dx          # ≈ 0.85
x_lif = x_center + dx          # ≈ 1.15


x_qif = 2          # ≈ 0.85
x_lif = 3          # ≈ 1.15

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

def pre_post(QIF, LIF):
    """
    Return a list that tells which output matrix (T, pre) belongs to each quadrant.
    """
    return [QIF, LIF, QIF, LIF]       # Q1,Q2,Q3,Q4 in that order


for dyn in dynamics:  # If not both dynamics, change above in parameters
    simulation_number = [i for i in range(1,5)] if dyn == "oscillations" else [i for i in range(9,13)]
    for q in quantifications:

        fig, axs = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(24, 8),          
        constrained_layout=True,
        sharex=False, sharey=True
        )

        for idx_pqif, pqif in enumerate(pqif_number):

            ax = axs[idx_pqif]  # ax[0], ..., ax[4]

            if pqif == 0:
                draw_lif = True
                draw_qif = False
                ax.set_title(f"pqif={pqif} - Only LIF")            
            elif pqif == 1:
                draw_lif = False
                draw_qif = True
                ax.set_title(f"pqif={pqif} - Only QIF")
            else:
                draw_lif = True
                draw_qif = True
                ax.set_title(f"pqif={pqif} - Heterogeneous")


            for idx, f in zip(simulation_number, slope):

                label = f

                # Path to connectivity matrix

                sim_path_mat = f"{base_path}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
                # sim_path_mat = f"{base_path}\\{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"
                # sim_path_mat = f"{dyn}\\simulation_{idx}\\simulation_{idx}_connectivity_matrix"


                # Output files

                sim_path_output = f"{base_path}\\simulation_{idx}\\simulation_{idx}_outputs"
                # sim_path_output = f"{base_path}\\{dyn}\\simulation_{sim}\\simulation_{sim}_outputs"

                # Containers that will hold per-seed mean activity for the two populations
                lif_means_per_seed = []
                qif_means_per_seed = []

                # Containers that will hold per-seed standard deviation for the two subpopulations
                lif_std_per_seed = []
                qif_std_per_seed = []


                for seed in seeds:


                    # Load the initialized matrix and the matrix after training (not used, but might be used later)
                    # path0 = f"{sim_path_mat}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_0_seed_{seed}"
                    # path11 = f"{sim_path_mat}\\simulation_{idx}_connectivity_pqif_{pqif}_iloop_11_seed_{seed}"
                    # W0 = load_matrix(path0)
                    # W11 = load_matrix(path11)
                    # mask_zero = np.isclose(W0, 0.0)
                    # W11_masked = np.where(mask_zero, np.nan, W11)
                    # Assign QIF and LIF as presynaptic neurons to correct quadrants
                    # presynaptic_neurons = pre_post(QIF, LIF)  # Becomes a list 
                    

                    # ------------------------------------------------------------

                    # Load output matrix
                    path_output = f"{sim_path_output}\\simulation_{idx}_outputs_pqif_{pqif}_iloop_11_seed_{seed}.csv"
                    output = load_matrix(path_output)

                    # Assign portions of output matrix to QIF and LIF variables
                    QIF, LIF = slice_neurons(output, pqif)


                    # ------------------------------------------------------------

                    # Compute the population mean (time‑averaged, then neuron‑averaged)
                    #   mean over time, shape (neurons,)
                    mean_time_QIF = np.nanmean(QIF, axis=0)   # size = n_QIF
                    mean_time_LIF = np.nanmean(LIF, axis=0)   # size = n_LIF

                    # mean over neurons, become a scalar for that config
                    pop_mean_QIF = np.nanmean(mean_time_QIF) if mean_time_QIF.size > 0 else np.nan
                    pop_mean_LIF = np.nanmean(mean_time_LIF) if mean_time_LIF.size > 0 else np.nan

                    qif_means_per_seed.append(pop_mean_QIF)
                    lif_means_per_seed.append(pop_mean_LIF)

                    # Compute the standard deviation on the time‑averaged array

                    # std over neurons, become a scalar for that config
                    pop_std_QIF = np.nanstd(mean_time_QIF, ddof=1) if mean_time_QIF.size > 0 else np.nan
                    pop_std_LIF = np.nanstd(mean_time_LIF, ddof=1) if mean_time_LIF.size > 0 else np.nan

                    qif_std_per_seed.append(pop_std_QIF)
                    lif_std_per_seed.append(pop_std_LIF)


                # -----------------------------------------------------------------

                # Collapse across seeds, to get mean +- std (error‑bars) of the MEAN 

                qif_means_per_seed = np.array(qif_means_per_seed, dtype=float)
                lif_means_per_seed = np.array(lif_means_per_seed, dtype=float)

                # Standard‑error is the sample standard deviation (ddof=1)
                qif_mean_mean  = np.nanmean(qif_means_per_seed) if qif_means_per_seed.size > 0 else np.nan
                qif_std_mean   = np.nanstd (qif_means_per_seed, ddof=1) if qif_means_per_seed.size > 0 else np.nan

                lif_mean_mean  = np.nanmean(lif_means_per_seed) if lif_means_per_seed.size > 0 else np.nan
                lif_std_mean   = np.nanstd (lif_means_per_seed, ddof=1) if lif_means_per_seed.size > 0 else np.nan
                

                # ------------------------------------------------------------

                # Collapse across seeds, to get mean +- std (error‑bars) of the STD 

                qif_std_per_seed = np.array(qif_std_per_seed, dtype=float)
                lif_std_per_seed = np.array(lif_std_per_seed, dtype=float)

                # Standard‑error is the sample standard deviation (ddof=1)
                qif_mean_std  = np.nanmean(qif_std_per_seed) if qif_std_per_seed.size > 0 else np.nan
                qif_std_std   = np.nanstd (qif_std_per_seed, ddof=1) if qif_std_per_seed.size > 0 else np.nan

                lif_mean_std  = np.nanmean(lif_std_per_seed) if lif_std_per_seed.size > 0 else np.nan
                lif_std_std   = np.nanstd (lif_std_per_seed, ddof=1) if lif_std_per_seed.size > 0 else np.nan

                # ------------------------------------------------------------


                # Collapse presynaptic activity across seeds

                # presyn_means_per_seed = np.array(presyn_means_per_seed) # convert to numpy
                # mean_presyn_per_quadrant = np.mean(presyn_means_per_seed, axis=0)  # means for each quadrant
                # stds_presyn_per_quadrant = np.std(presyn_means_per_seed, axis=0, ddof=1)  # std to get errorbars



                # ------------------------------------------------------------

                # Formatting

                # if pqif == 1:
                #     # color = 'steelblue'

                #     if idx != simulation_number[0]:  
                #         # only take the first instance of QIF, since homogenous QIF does not change per simulation outside of variance
                #         # also to not have more seeds for QIF than the other cases
                #         continue 

                col=(color_map_slope[f] if pqif != 1 else 'steelblue')
                # y = mean_presyn_per_quadrant
                # yerror = stds_presyn_per_quadrant
                # label = f"{f}"

                if dyn == "oscillations":
                    fillstyle='full'
                else:
                    fillstyle='none'



                # ------------------------------------------------------------

                # Plotting

                # ---- QIF -------------------------------------------------------
                # if draw_qif:
                #     # Only the first simulation should appear 
                #     label_qif = 'QIF' if idx == simulation_number[0] else None
                #     ax.errorbar(
                #         x_qif,
                #         qif_mean_mean,
                #         yerr=qif_std_mean,
                #         fmt='o',
                #         capsize=5,
                #         color='steelblue',
                #         elinewidth=1.2,
                #         markersize=10,
                #         fillstyle=fillstyle,
                #         label=label_qif,
                #     )

                # ---- LIF -------------------------------------------------------
                if draw_lif:
                    # Legend entry only once per slope (and only for the mixed panels)
                    label_lif = (f'{f}' if (pqif not in (0, 1) and idx == simulation_number[0])
                                 else None)
                    ax.errorbar(
                        x_lif,
                        lif_mean_mean,
                        yerr=lif_std_mean,
                        fmt='o',
                        capsize=5,
                        color=col,
                        elinewidth=1.2,
                        markersize=10,
                        fillstyle=fillstyle,
                        label=label_lif,
                    )

                    ax.axhline(lif_mean_mean, color=col, alpha=0.2)

            # QIF outside
            if draw_qif:
                    # Only the first simulation should appear 
                    label_qif = 'QIF' if idx == simulation_number[0] else None
                    ax.errorbar(
                        x_qif,
                        qif_mean_mean,
                        yerr=qif_std_mean,
                        fmt='o',
                        capsize=5,
                        color='steelblue',
                        elinewidth=1.2,
                        markersize=10,
                        fillstyle=fillstyle,
                        label=label_qif,
                    )

                    ax.axhline(qif_mean_mean, color='steelblue', alpha=0.2)

                # if idx == simulation_number[0]:
                #     ax.errorbar(
                #         x_qif,                                        # x‑position for QIF
                #         y[0],                                     # mean value (QIF)
                #         yerr=yerror[0],                           # std‑error (QIF)
                #         fmt='o',
                #         capsize=5,
                #         color='steelblue',                        # QIF is always steel‑blue
                #         elinewidth=1.2,
                #         markersize=10,
                #         fillstyle=fillstyle,
                #         label='QIF'                               # legend only for first sim
                #     )
                
                # ax.errorbar(
                #     x_lif,                                            # x‑position for LIF
                #     y[1],                                         # mean value (LIF)
                #     yerr=yerror[1],                               # std‑error (LIF)
                #     fmt='o',
                #     capsize=5,
                #     color=color_map_slope[f],
                #     elinewidth=1.2,
                #     markersize=10,
                #     fillstyle=fillstyle,
                #     label=f'{f}' if (pqif not in (0, 1) and idx == simulation_number[0]) else None
                # )
                
                # ax.errorbar(quadrant_names[:2], y[:2], fmt='o', yerr=yerror[:2], capsize=5, color=color, elinewidth=1.2, markersize=6, label=label)
                    

            ax.set_xlabel("Presynaptic neuron")
            ax.set_title(f"pqif: {pqif}")
            ax.set_box_aspect(1)
            ax.set_xlim(1, 4)
            yticks = np.arange(start=0.18, stop=0.62, step=0.1)
            ax.set_ylim(0.18, 0.62)
            ax.set_yticks(yticks)

            if idx_pqif == 0:
                ax.set_ylabel("r_j (mean +- std)")
        fig_path = FIGURES_DIR / f"{dyn}_output_subpopulations.svg"
        plt.savefig(fig_path, dpi=300)
        print(f"Figure saved in '{fig_path}'")
        # plt.suplots_adjust(hspace=None)
        plt.suptitle(f"{dyn.capitalize()}: Mean +- $\sigma$ of output of presynaptic neuron", fontsize=20, weight='bold')
        plt.show()
        plt.legend()




# Animation (TEST)
# import matplotlib.animation as animation

# fig, ax = plt.subplots(figsize=(8, 4))
# line, = ax.plot([], [], lw=2)
# ax.set_xlim(0, df.shape[1])
# ax.set_ylim(df.values.min(), df.values.max())
# ax.set_xlabel('Neuron')
# ax.set_ylabel(r'$r_j(t)$')
# ax.set_title('Current snapshot over time')

# def init():
#     line.set_data([], [])
#     return line,

# def animate(frame):
#     y = df.iloc[frame].values
#     line.set_data(np.arange(len(y)), y)
#     ax.set_title(f'Current at time step {frame}')
#     return line,

# ani = animation.FuncAnimation(fig, animate, frames=df.shape[0],
#                               init_func=init, blit=True, interval=30)
# # Save as GIF or MP4 if you like:
# # ani.save('current_movie.mp4', writer='ffmpeg')
# plt.show()

