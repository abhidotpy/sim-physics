"""
Python program to simulate the Ising model for N x N spins.
It also calculates magnetization for no reason whatsoever.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmp
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation


# Defining constants for our simulation
N = 100                                           # No. of spins each side (N x N grid of spins)
frame_count = 100                                 # No. of frames for the animation
DT = 4 / frame_count                              # Temperature drops from 4 to 0.01 in frame_count number of frames
T = np.array([4.01 - i * DT for i in range(frame_count)])   # Array of temperatures
beta = 1 / T                                      # beta = 1/kT, where Boltzmann constant k = 1
Z = 2 * np.random.randint(2, size=(N, N, 1)) - 1  # Initial random spin state
M = np.array(np.sum(Z[:, :, 0]) / N**2)           # Magnetization

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
artist = []


def init_axis():        # Initialize the axis parameters
    ax2.set_xlabel(r'Temperature $\rightarrow$')
    ax2.set_ylabel(r'Magnetization $\rightarrow$')
    ax2.set_xlim([0, 4])                        # Temperature drops from 4 to zero
    ax2.set_ylim([-0.5, 0.5])                   # Magnetization plot limits
    ax2.axhline(color='chocolate', alpha=0.4)   # Axis line


def init_artist():      # Create the display artists
    global artist
    im = ax1.imshow(Z[:, :, 0], cmap=cmp.binary, extent=(0, 1, 0, 1), interpolation='nearest', animated=True)
    pm, = ax2.plot(T[0], M, animated=True)

    artist.append(im)
    artist.append(pm)


def ising():
    """
    The main loop which pre-calculates the spin state each frame for display. The spin states for each
    frame is stored in third axis of Z array. This is done to make animate loop run faster.
    """
    global Z, M
    for i in range(frame_count):
        print('Frame :', i)
        del_E = 0
        Y = Z[:, :, -1].reshape(N, N, 1)
        for j in range(N ** 2):
            a, b = np.random.randint(0, N, 2)
            spin = Y[a, b]          # spin at a point
            ns = Y[a + 1 if a < N - 1 else N - a - 1, b] + Y[a - 1 if a > 0 else N - a - 1, b] \
                 + Y[a, b + 1 if b < N - 1 else N - b - 1] + Y[a, b - 1 if b > 0 else N - b - 1]
            dE = 2 * spin * ns

            if dE < 0:      # Does change of spin lower the energy
                spin = spin * -1
                del_E = del_E + spin
            else:
                if np.random.rand() < np.exp(-beta[i] * dE):    # Probability from MB distribution
                    spin = spin * -1
                    del_E = del_E + spin

            Y[a, b] = spin

        Z = np.concatenate((Z, Y), axis=2)              # Append frame data on third axis
        M = np.append(M, np.sum(Z[:, :, i]) / N**2)     # Calculate average magnetization each spin
    M = gaussian_filter1d(M, 3)


def animate(frame):         # Animation mainloop
    global artist
    artist[0].set_data(Z[:, :, frame])
    artist[1].set_data(T[:frame], M[:frame])

    return artist


init_axis()
init_artist()
ising()

aaaa = FuncAnimation(fig, animate, frames=frame_count, interval=30, blit=True)

plt.show()

