# ------------------------------------------------------------
#  Script to plot output current
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


# Python script
import pathlib, os
plt.style.use(os.path.join(pathlib.Path(__file__).parent, 'plos.mplstyle'))

# Jupyter script
# style_path = Path.cwd() / "plos.mplstyle"   # cwd = folder Jupyter launched from
# plt.style.use(str(style_path))

# ------------------------------------------------------------

# Color map

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

# Make colors once
color_map_vrest = make_color_map(v_rest, plt.cm.Reds)  # Reds for vrest
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

# File


# File

sim=9
pqif=0.5
iloop=11

quadrant_names = ["Q1", "Q2", "Q3", "Q4"]


vrest = [-8.5, -12.3, -17, -22]
slope = [14.44, 10.68, 8.65, 7.18]
simulation_number = [i for i in range(1,5)]
slope_qif = 10.74
all_slopes = [14.44, 10.68, 10.74, 8.65, 7.18]
pqif_number = [0, 0.25, 0.5, 0.75, 1]
pqif_number = [0.5]

quantifications = ["mean", "std"]
# dynamics = ["sequences"]  
# dynamics = ["oscillations", "sequences"]
dynamics = ["oscillations"]

seeds = range(2)

def load(path):
    return np.genfromtxt(path, delimiter=',')

def slice_neurons(output, pqif):
    '''
    Slice output based on pqif. First pqif*N are QIF and remaining LIF.
    '''

    N = output.shape[0]


    h = int(round(pqif * N))

    # All rows (time), QIF and LIF
    QIF = output[:, :h]  # up to QIF
    LIF = output[:, h:]  # from LIF


    return QIF, LIF

for dyn in dynamics:
    simulation_number = [i for i in range(1,5)] if dyn == "oscillations" else [i for i in range(9,13)]
    for pqif in pqif_number:

        for idx, f in zip(simulation_number, slope):

            mean = []
            std = []

            for seed in seeds:
                # qif_r = 


                simulation_path = f"{base_path}\\simulation_{idx}\\simulation_{idx}_outputs"
                # simulation_path = f"{base_path}\\{dyn}\\simulation_{idx}\\simulation_{idx}_outputs"

                path = f"{simulation_path}\\simulation_{idx}_outputs_pqif_{pqif}_iloop_11_seed_{seed}.csv"

                # df = pd.read_csv(path, header=None)


                output = load(path)

                QIF, LIF = slice_neurons(output, pqif)

                # TODO This is not finished



# Animation
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, df.shape[1])
ax.set_ylim(df.values.min(), df.values.max())
ax.set_xlabel('Neuron')
ax.set_ylabel(r'$r_j(t)$')
ax.set_title('Current snapshot over time')

def init():
    line.set_data([], [])
    return line,

def animate(frame):
    y = df.iloc[frame].values
    line.set_data(np.arange(len(y)), y)
    ax.set_title(f'Current at time step {frame}')
    return line,

ani = animation.FuncAnimation(fig, animate, frames=df.shape[0],
                              init_func=init, blit=True, interval=30)
# Save as GIF or MP4 if you like:
# ani.save('current_movie.mp4', writer='ffmpeg')
plt.show()

