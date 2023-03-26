"""
Python program to simulate the fallout of particles from a nuclear reactor as a
random walk problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy.random as rd
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Defining constants
Ymax = 10                           # Height of the simulation box
X0, X1, X2, X3 = 0, 5, 10, 15       # Coordinates of partitions for walls
STEP = 0.2                          # Step size for particle motion
RADIUS = 0.1                        # Radius of particles
NO_OF_PARTICLES = 1000              # Total number of particles used
DIR = tuple([(1, 0, 3), (0, 1, 2), (1, 2, 3), (0, 3, 2)])  # Available directions
COUNT = np.zeros(3)                 # Array to store number of particles in each region
rand = rd.default_rng()             # Initialize the random number generator
particle = []
artist = []
xx, yy = [], []

fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)


class ParticleGenerator:
    def __init__(self, id_):
        x = rd.uniform(0, 2)
        y = rd.uniform(0, 10)
        self.pos = np.array([x, y])     # Particle starts at a random coordinate
        self.id = id_
        self.life = 100                 # Particle survives for 100 random walks
        self.velocity = 0               # Particle has zero initial velocity


def init_particle():        # Create the particles
    for i in range(NO_OF_PARTICLES):
        particle.append(ParticleGenerator(i))


def init_canvas():      # Create the background and regions
    plt.xlim([0, X3])
    plt.ylim([0, 10])
    plt.title('Nuclear Reactor')
    plt.xticks([])
    plt.yticks([])
    ax.add_patch(Rectangle((0, 0), X2 - X1, Ymax, facecolor='yellow', alpha=0.2))
    ax.add_patch(Rectangle((X1, 0), X2 - X1, Ymax, facecolor='gray', alpha=0.2))
    ax.add_patch(Rectangle((X2, 0), X2 - X1, Ymax, facecolor='turquoise', alpha=0.2))
    plt.vlines([X1, X2], 0, Ymax, colors=('k', 'k'))
    plt.text(2.5, 9.5, 'Reactor', size='large', ha='center', bbox=dict(facecolor='w'))
    plt.text(7.5, 9.5, 'Shield', size='large', ha='center', bbox=dict(facecolor='w'))
    plt.text(12.5, 9.5, 'Air', size='large', ha='center', bbox=dict(facecolor='w'))
    fig.canvas.draw()


def init_artists():     # Create the display objects
    global artist
    artist = [plt.text(0.1, 0.1, '', bbox=dict(facecolor='w'), animated=True),
              plt.text(2.5, 0.1, '', size='large', ha='center', bbox=dict(facecolor='w'), animated=True),
              plt.text(7.5, 0.1, '', size='large', ha='center', bbox=dict(facecolor='w'), animated=True),
              plt.text(12.5, 0.1, '', size='large', ha='center', bbox=dict(facecolor='w'), animated=True)]

    for pt in particle:
        c = Ellipse(pt.pos, width, height, facecolor='black', animated=True)
        artist.append(c)
        ax.add_patch(c)


def animate(frame):
    global artist, COUNT
    COUNT = np.zeros(3)
    for pt in particle:

        if pt.pos[0] < X1:
            COUNT[0] += 1
        elif pt.pos[0] < X2:
            COUNT[1] += 1
        else:
            COUNT[2] += 1

        if X1 < pt.pos[0] < X2:
            pt.velocity = rand.choice(DIR[pt.velocity], p=[0.25, 0.5, 0.25])
            if pt.life <= 0:
                continue

        if pt.velocity & 1:
            pt.pos[1] = pt.pos[1] + STEP * (1 - 2 * (pt.velocity == 3))
            pt.pos[1] += Ymax if pt.pos[1] < 0 else 0
            pt.pos[1] -= Ymax if pt.pos[1] > Ymax else 0
        else:
            pt.pos[0] = pt.pos[0] + STEP * (1 - 2 * (pt.velocity == 2))

        artist[pt.id + 4].set_center(pt.pos)

        pt.life = pt.life - 1

    artist[0].set_text('Frame : {}'.format(frame))
    artist[1].set_text(str(int(COUNT[0])))
    artist[2].set_text(str(int(COUNT[1])))
    artist[3].set_text(str(int(COUNT[2])))
    return artist


init_canvas()
init_particle()
init_artists()

aa = FuncAnimation(fig, animate, frames=200, interval=30, blit=True)

plt.show()
