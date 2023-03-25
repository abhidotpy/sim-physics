"""
Python code to draw a figure using complex Fourier Transform. The shape is supplied as an array
of (x, y) coordinates as a single closed curve.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

L = 5                       # Size of display canvas
DT = 0.01                   # Timestep of animation
F_RANGE = 300               # No. of frequencies used for the plot
path_x, path_y = [], []     # Store the coordinates of drawn path
artist = []                 # Drawable artist array

df = pd.read_csv('D:\\Python\\PyCharm Projects\\Design CSV\\Music Note.csv')
X = np.array(df.iloc[:, 0])     # x coordinates of the figure points
Y = np.array(df.iloc[:, 1])     # y coordinates of the figure points
Z = X + Y*1j                    # Recast them as a closed circle in complex plane

F = fft(Z, norm='forward')          # Fourier transform of the complex figure
freq = np.arange(-F_RANGE, F_RANGE + 1)                 # Frequencies used for the construction
amp = np.concatenate((F[-F_RANGE:], F[:F_RANGE + 1]))   # Their respective amplitudes obtained from fft
N = len(amp)
G = sorted(zip(amp, freq), key=lambda x: abs(x[0]), reverse=True)
amp = np.array(list(map(lambda x: x[0], G)))        # Sorted amplitudes in descending order
freq = np.array(list(map(lambda x: x[1], G)))       # The respective frequencies also sorted

fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)


def init_axes():    # Initialize the axis parameters
    plt.xlim([-L, L])
    plt.ylim([-L, L])
    plt.xticks([])
    plt.yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.gca().set_aspect('equal')


def init_artist():      # Initialize the drawable objects
    ln, = plt.plot(0, 0, '.-', c='teal', animated=True)
    path, = plt.plot(0, 0, 'k', animated=True)
    artist.append(ln)
    artist.append(path)

    for i, f in enumerate(amp):
        c = Circle((0, 0), abs(f), edgecolor='orange', fill=None, alpha=0.7, animated=True)
        ax.add_patch(c)
        artist.append(c)


def animate(frame):
    t = frame * DT

    P = np.cumsum(amp * np.exp(1j * freq * t))  # Construct the brush line
    P = np.insert(P, 0, complex(0, 0))          # Brush line should start at (0, 0)

    path_x.append(P[-1].real)
    path_y.append(P[-1].imag)

    artist[0].set_data(P.real, P.imag)    # Animate the brush line
    artist[1].set_data(path_x, path_y)    # Pattern drawn by the brush

    for i, c in enumerate(artist[2:]):
        c.set_center((P[i].real, P[i].imag))    # Change circle position based on brush

    return artist

init_axes()
init_artist()

aaaa = FuncAnimation(fig, animate, frames=1000, repeat=False, interval=30, blit=True)

plt.get_current_fig_manager().window.showMaximized()
plt.show()

