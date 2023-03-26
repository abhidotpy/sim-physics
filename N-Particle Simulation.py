from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Defining constants of system
L = 5       # Size of animation box
B = 1       # Number of blocks
V = 5
PARTICLE_NUMBER = 5
DT = 0.01
RADIUS = 0.2

particle: List[Any] = []
block_index = []
block_particle = []
artists = []
fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)


class ParticleGenerator:
    def __init__(self, id, particle_position, particle_velocity, particle_radius, block_id):
        self.id = id
        self.pos = np.array(particle_position)
        self.vel = np.array(particle_velocity)
        self.rad = particle_radius
        self.blk_id = block_id


def init_blocks():
    global block_index, block_particle
    no_of_blocks = B ** 2
    block_index = np.array(range(no_of_blocks)).reshape(B, B)
    block_index = tuple(block_index.T)

    block_particle = [[] for i in range(no_of_blocks)]


def init_particle():
    for i in range(PARTICLE_NUMBER):
        x_pos = np.random.uniform(0, L)
        y_pos = np.random.uniform(0, L)
        theta = np.random.uniform(0, 2 * np.pi)
        x_vel = V * np.cos(theta)
        y_vel = V * np.sin(theta)

        d1, d2 = np.floor_divide([x_pos, y_pos], [L / B, L / B])
        d1 = d1 - 1 if x_pos > L else d1
        d2 = d2 - 1 if y_pos > L else d2
        current_block = block_index[int(d1)][int(d2)]

        p = ParticleGenerator(i, [x_pos, y_pos], [x_vel, y_vel], RADIUS, current_block)
        particle.append(p)
        block_particle[current_block].append(particle[i].id)


def init_axis():
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.vlines([1, 2, 3, 4], 0, 5, alpha=0.3)
    # ax.hlines([1, 2, 3, 4], 0, 5, alpha=0.3)


def init_artist():
    tex = plt.text(0.1, 0.1, '', bbox=dict(facecolor='w'), animated=True)
    artists.append(tex)

    for p in particle:
        C = Circle(p.pos, p.rad, edgecolor='k', facecolor='turquoise', animated=True)
        ax.add_patch(C)
        artists.append(C)


def wall_collision(i):
    pos_x, pos_y = particle[i].pos
    r = particle[i].rad

    if L - pos_x < r:  # Right Wall
        particle[i].vel[0] = -particle[i].vel[0]
        particle[i].pos[0] = 2 * (L - r) - particle[i].pos[0]

    elif pos_x < r:  # Left Wall
        particle[i].vel[0] = -particle[i].vel[0]
        particle[i].pos[0] = 2 * r - particle[i].pos[0]

    elif L - pos_y < r:  # Top Wall
        particle[i].vel[1] = -particle[i].vel[1]
        particle[i].pos[1] = 2 * (L - r) - particle[i].pos[1]

    elif pos_y < r:  # Bottom Wall
        particle[i].vel[1] = -particle[i].vel[1]
        particle[i].pos[1] = 2 * r - particle[i].pos[1]


def particle_collision():
    for b in block_particle:
        g = np.meshgrid(b, b, indexing='ij')
        col_pairs = list(filter(lambda x: x[0] != x[1], zip(np.ravel(np.triu(g[0])), np.ravel(np.triu(g[1])))))
        for j in col_pairs:
            v1 = particle[j[0]].vel
            v2 = particle[j[1]].vel

            dis = particle[j[0]].pos - particle[j[1]].pos   # Distance vector between two particles
            dis_len = np.linalg.norm(dis)                   # Norm of distance vector
            dis_unit = dis / dis_len                        # Unit distance vector

            if dis_len <= (2.1 * RADIUS):
                v_comp_mag = np.dot(v2, dis_unit)
                v_comp = v_comp_mag * dis_unit
                particle[j[0]].vel = v1 + v_comp
                particle[j[1]].vel = v2 - v_comp

                particle[j[0]].pos = particle[j[0]].pos + (1/2.0)*(2*RADIUS - dis_len)*dis_unit
                particle[j[1]].pos = particle[j[1]].pos - (1/2.0)*(2*RADIUS - dis_len)*dis_unit


def animate(frame):
    global artists
    for p in particle:
        block_old = p.blk_id

        p.pos = p.pos + p.vel * DT
        wall_collision(p.id)

        d1, d2 = np.floor_divide(p.pos, [L / B, L / B])
        d1 = d1 - 1 if p.pos[0] > L else d1
        d2 = d2 - 1 if p.pos[1] > L else d2
        block_new = block_index[int(d1)][int(d2)]
        p.blk_id = block_new

        if block_old != block_new:
            block_particle[block_old].remove(p.id)
            block_particle[block_new].append(p.id)

        artists[p.id + 1].set_center(p.pos)

    particle_collision()
    artists[0].set_text('Frame : {}'.format(frame))
    return artists


init_blocks()
init_particle()
init_axis()
init_artist()


aaaa = FuncAnimation(fig, animate, frames=1000, interval=30, blit=True)

plt.show()


