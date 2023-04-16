"""
Python program to simulate the magnetic pendulum fractal. The pendulum is released from NxN initial conditions
and is allowed to swing under the influence of three magnets. The pixel from which pendulum is released is colored
based on the closest magnet to it after T=100s.
"""

import numpy as np
from scipy.integrate import odeint
from numpy.linalg import norm
import matplotlib.pyplot as plt


def f(x0, t0, K=1, b=0.1, KM=1, e=0.2):
    """
    The main derivative function for odeint
    :param x0: The (pos_x, pos_y, vel_x, vel_y) array
    :param t0: The independent time variable
    :param K: The restoring force constant for the pendulum
    :param b: The damping coefficient
    :param KM: The force constant for magnetic attraction
    :param e: The height of pendulum from the ground
    """
    r, v = np.array(x0[:2]), np.array(x0[2:])       # Extract position and velocity
    acc_mag = sum([r - r_mag / np.power(norm(r - r_mag) ** 2 + e ** 2, (3 / 2)) for r_mag in P_mag])
    ac_x, ac_y = -(K*r) - (b*v) - (KM * acc_mag)    # Calculate acceleration xy components
    return np.array([x0[2], x0[3], ac_x, ac_y])     # Return in dxdt, dydt, dvxdt, dvydt


# Defining constants
T = np.linspace(0, 100, 101)        # Time points for integration
N = 100                             # No of initial conditions
L = 3                               # Size of the domain
R = 1                               # Distance of magnets from center
pts = np.linspace(-L, L, N)         # Dividing the domain into N points
col_array = np.array([1, 2, 3])     # Index of each magnet
x_init = np.meshgrid(pts, pts, indexing='ij')   # Each domain point is an initial condition
P_mag = np.array([[R * np.cos(d), R * np.sin(d)] for d in np.radians([-30, -150, -270])])  # Magnet positions
C = np.zeros([N**2])                # Initialize the points
points = zip(np.ravel(x_init[0]), np.ravel(x_init[1]))


# Setting up the plotting figure
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.xlim([-L, L])
plt.ylim([-L, L])
plt.gca().set_aspect('equal')


# The main loop that calculates the trajectories of the magnetic pendulum
for i, x0 in enumerate(points):
    print(i)
    X = odeint(f, [*x0, 0, 0], T, args=(0.2, 0.15, 1, 0.5))
    final_pos = np.repeat([X[-1, 0:2]], 3, axis=0)
    m_low = np.argmin(norm(final_pos - P_mag, axis=1))
    C[i] = col_array[m_low]     # Point value set to index of nearest magnet


# Plotting the final fractal pattern
C = C.reshape(N, N)
plt.imshow(C, cmap=plt.cm.brg, extent=(-3, 3, -3, 3))

plt.show()

