import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation


class Ball:
    def __init__(self, idx, position, velocity, col_idx):
        self.id = idx
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        self.color = cm.jet(col_idx)
        self.path_x = []
        self.path_y = []


DT = 0.1                                    # Timestamps for the trajectories
sep = 1e-8                                  # Initial separation between balls
G = 0.1                                     # Acceleration die to gravity
Nb = 100                                    # Number of balls
bounces = 50                                # Maximum number of bounces
col_array = np.linspace(0.2, 0.9, Nb)       # Colors of the balls
ball_array = []                             # For storing the ball objects
path_length = []                            # Store the length of the path for each ball
trajectory = []                             # Line object array for plotting the trajectories

# Axes creation and formatting
fig, ax = plt.subplots()
fig.set_facecolor('k')
ax.set_facecolor('k')
plt.subplots_adjust(0.155, 0.007, 0.845, 0.998, hspace=0.04)
plt.gca().set_aspect('equal')
plt.xticks([])
plt.yticks([])
plt.ylim([-1.1, 0.5])
ax.spines[:].set_visible(False)


# Create the balls
for i in range(Nb):
    bb = Ball(i+1, [0.5 + i * sep, 0], [0, 0], col_array[i])
    ball_array.append(bb)


# Draw a circle on which balls fall
xx = np.linspace(-1, 1, 10000)
yy = -np.sqrt(1 - xx ** 2)
plt.plot(1.02 * xx, 1.02 * yy, c='white')


def find_path(pos_initial, vel_initial):
    pos_new, vel_new = pos_initial, vel_initial
    Xp, Yp = [], []

    def ev(t, x):
        return x[0] ** 2 + x[1] ** 2 - 1

    ev.terminal = True

    def f(t, x):
        return x[2], x[3], 0, -G

    # Path calculation
    for j in range(bounces):

        # Find path by solving a projectile motion
        Y = solve_ivp(f, [0, 100], [*pos_new, *vel_new], events=ev, dense_output=True)
        Np = int(Y.t[-1] / DT)
        T = np.linspace(0, Y.t[-1], Np)
        Z = Y.sol(T)
        Xp.extend(Z[0])
        Yp.extend(Z[1])

        # Adjust the velocity vector at point of collision
        surface_normal = Z[:2, -1]
        surface_normal = surface_normal / norm(surface_normal)
        projection_matrix = np.eye(2) - 2 * np.outer(surface_normal, surface_normal)
        vel_new = projection_matrix @ Z[2:, -1]
        pos_new = Z[:2, -1] + vel_new * 1e-6 - (1 / 2) * G * 1e-12

    return Xp, Yp


# Calculate the trajectory of each ball
for bb in ball_array:
    print(bb.id)
    bb.path_x, bb.path_y = find_path(bb.pos, bb.vel)
    path_length.append(len(bb.path_x))
    l, = plt.plot([0], [0], 'o', markersize=5, c=bb.color, animated=True)
    trajectory.append(l)


tot_frames = min(path_length)
print(min(path_length), '\n')
    

def animate(frame):
    # print(frame)
    global ball_array, path_length, trajectory
    for ball, line in zip(ball_array, trajectory):
        line.set_data([ball.path_x[frame]], [ball.path_y[frame]])

    return trajectory


aaaa = FuncAnimation(fig, animate, frames=tot_frames, interval=30, repeat=False, blit=True)

plt.get_current_fig_manager().window.showMaximized()
plt.show()

