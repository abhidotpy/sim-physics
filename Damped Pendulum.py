"""
Python code to study the motion of a pendulum for different potentials, and hence
different degrees of restoring forces.
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

THETA = pi/3                    # Initial angle of pendulum
W = 1                           # Natural frequency of oscillation
B = 0                           # Damping factor
DT = 0.1                        # Timestamps for animation
dtheta = np.array([THETA, 0])   # Array to store evolution of theta and d(theta)
TIME = []                       # Time points array
TH_ARRAY, VL_ARRAY = [], []     # Phase plot points array
artist = []                     # Drawable artist array

# Create the figure object and the axes
fig = plt.figure()
ax1 = fig.add_axes([0.02, 0.05, 0.45, 0.90], aspect='equal')
ax2 = fig.add_axes([0.53, 0.65, 0.45, 0.3])
ax3 = fig.add_axes([0.53, 0.08, 0.45, 0.5])

def f(t, x):
    """
    The main derivative function dx/dt = f(t, x)
    """
    return np.array([x[1], -2 * B * x[1] - W * W * np.sin(x[0])])


def RK4(func, t0, x0, h):
    """
    The fourth-order Runge-Kutta method
    """
    k1 = func(t0, x0)
    k2 = func(t0 + h / 2, x0 + 0.5 * h * k1)
    k3 = func(t0 + h / 2, x0 + 0.5 * h * k2)
    k4 = func(t0 + h, x0 + h * k3)
    return x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def init_axes():
    """
    Initializing the axes for display

    ax1: The main pendulum display axes.
    ax2: The axis displaying the sine graph.
    ax3: The axis for the phase plot of pendulum.
    """
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])

    ax2.set_xlim([0, 30])
    ax2.set_ylim([-pi/2, pi/2])
    ax2.set_xlabel('t')
    ax2.set_ylabel('x(t)')
    ax2.grid(True)

    ax3.set_xlim([-pi/2, pi/2])
    ax3.set_ylim([-pi/2, pi/2])
    ax3.set_xlabel('x(t)')
    ax3.set_ylabel('v(t)')
    ax3.grid(True)


def init_artist():
    """
    Initializing the artist objects for display
    """
    init_theta = THETA, 0

    string = Rectangle((0, 0.5), 1, 0.01, angle=-90, color='k', animated=True)
    bob = Circle((np.sin(pi), 0.5+np.cos(pi)), 0.07, color='k', animated=True)
    line_xt, = ax2.plot(0, 0, color='b', animated=True)
    phase_point = Circle(init_theta, 0.02, color='k', animated=True)
    line_phase, = ax3.plot(init_theta, color='g', animated=True)

    ax1.add_patch(string)
    ax1.add_patch(bob)
    ax3.add_patch(phase_point)

    artist.extend([string, bob, line_xt, phase_point, line_phase])


def animate(frame):
    """
    Main animation function
    """
    global dtheta
    string, bob, line_plt, phase_point, phase_plt = artist

    # Update next point based on differential equation
    ang = dtheta[0]
    t_next = frame * DT
    dtheta = RK4(f, t_next, dtheta, DT)
    TIME.append(t_next)
    TH_ARRAY.append(dtheta[0])
    VL_ARRAY.append(dtheta[1])

    # Change artist property on canvas
    string.set_angle(-90 + np.degrees(ang))
    bob.set_center((np.sin(pi - ang), 0.5 + np.cos(pi - ang)))
    line_plt.set_data(TIME, TH_ARRAY)
    phase_point.set_center((dtheta[0], dtheta[1]))
    phase_plt.set_data(TH_ARRAY, VL_ARRAY)

    return artist


init_axes()
init_artist()

aaaa = FuncAnimation(fig, animate, frames=1000, interval=30, blit=True)

plt.show()
