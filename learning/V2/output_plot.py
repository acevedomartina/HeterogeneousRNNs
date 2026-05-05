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

ONEDRIVE_BASE = Path(r"C:\Users\Silje\OneDrive\Dokumenter\mscneuroscience20242026\nevr3901\simulations_folder\may_simulations")
ONEDRIVE_BASE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------

# File

sim=9
pqif=0.5
iloop=11
df = pd.read_csv(ONEDRIVE_BASE / f"simulation_{sim}/simulation_{sim}_outputs/simulation_{sim}_outputs_pqif_{pqif}_iloop_{iloop}_seed_0.csv", header=None)

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

