'''
Python program to simulate the chaotic nature of a system of two balls falling under gravity
on a half circle of radius one. All collisions are elastic.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


DT = 0.05                                   # Timestamps
rad = 0.02                                  # Radius of ball
del_0 = 1e-4                                # Initial separation of position
tot_frames = 2000                           # Total number of frames
G = np.array([0, -0.2])                     # Acceleration due to gravity
pos1 = np.array([0.7, 0])                   # Initial position of blue ball
pos2 = np.array([0.7 + del_0, 0])           # Initial position of red ball
vel1 = np.zeros_like(pos1)                  # Initial velocity of blue ball
vel2 = np.zeros_like(pos2)                  # Initial velocity of red ball
lya_exp = np.array([[0, np.log10(del_0)]])  # Lyapunov exponent graph points


# Formatting the axes
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
ax1.set_ylim([-1.2, 0.8])
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_ylim([np.log10(del_0)-1, 1])
ax2.set_xlim([0, tot_frames * DT])


# Create the falling balls
C1 = Circle(pos1, radius=rad, edgecolor='k', facecolor='turquoise', animated=True)
C2 = Circle(pos2, radius=rad, edgecolor='k', facecolor='orangered', animated=True)
ax1.add_patch(C1)
ax1.add_patch(C2)

# Draw the graph for Lyapunov exponent
lm, = ax2.plot(lya_exp[:, 0], lya_exp[:, 1], 'g', animated=True)
ax2.set_title('Lyapunov Exponent')


# Draw the curve on which the balls fall
theta = np.linspace(0, -np.pi, 100)
xx, yy = np.cos(theta), np.sin(theta)
ax1.plot(xx, yy, 'k')
ax1.spines[['left', 'top', 'right', 'bottom']].set_visible(False)


def animate(frame):
    print(frame)
    global pos1, pos2, vel1, vel2, lya_exp

    # Update position and velocity based on Newton's equations
    pos1 = pos1 + vel1 * DT + (1 / 2) * G * DT * DT
    pos2 = pos2 + vel2 * DT + (1 / 2) * G * DT * DT
    vel1 = vel1 + G * DT
    vel2 = vel2 + G * DT

    mag_pos1 = norm(pos1)
    mag_pos2 = norm(pos2)

    if mag_pos1 + rad > 1:      # Blue ball collision test
        surf_norm = pos1 / mag_pos1
        proj_matrix = np.eye(2) - 2 * np.outer(surf_norm, surf_norm)
        vel1 = proj_matrix @ vel1
        pos1 = pos1 - rad * surf_norm

    if mag_pos2 + rad > 1:      # Red ball collision test
        surf_norm = pos2 / mag_pos2
        proj_matrix = np.eye(2) - 2 * np.outer(surf_norm, surf_norm)
        vel2 = proj_matrix @ vel2
        pos2 = pos2 - rad * surf_norm

    C1.set_center(pos1)     # Update position of blue ball
    C2.set_center(pos2)     # Update positin of red ball

    # Calculation for Lyapunov exponent
    x1 = lya_exp[-1][0] + DT
    y1 = np.log10(norm(pos2 - pos1))
    lya_exp = np.vstack([lya_exp, [x1, y1]])
    lm.set_data(lya_exp[:, 0], lya_exp[:, 1])

    return C1, C2, lm


aaaa = FuncAnimation(fig, animate, frames=tot_frames, interval=30, repeat=False, blit=True)

plt.show()
