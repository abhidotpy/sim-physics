"""
Python program to estimate the value of pi using the Needle Drop experiment
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons

# Defining constants
PI = 0                  # Estimated value of pi
D = 1                   # Length of a needle
WIDTH = D
HEIGHT = 0.01
NEEDLE_COUNT = 0        # Counter for total no of needles
INTER_COUNT = 0         # Counter for no of intersecting needles
MUL = 1                 # Multiplier for dropping needles
X = [NEEDLE_COUNT]      # Store the total number of needles dropped
Y = [PI]                # Store the estimated value of pi each drop


class Needle:
    def __init__(self):
        self.x = rd.uniform(0, 10)
        self.y = rd.uniform(0, 10)
        self.width = WIDTH
        self.height = HEIGHT
        self.angle = rd.uniform(0, 2 * np.pi)
        self.x2 = self.x + self.width * np.cos(self.angle)
        self.y2 = self.y + self.width * np.sin(self.angle)

    def draw_needle(self, axis):
        ang_deg = np.degrees(self.angle)
        r = Rectangle([self.x, self.y], self.width, self.height, angle=ang_deg)
        r.set_color('black')
        axis.add_patch(r)

    def is_intersecting(self):
        m = np.floor(self.y)
        n = np.floor(self.y2)
        if m != n:
            return True
        else:
            return False


# Create the plotting area
fig = plt.figure()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
ax = plt.axes([0.05, 0.05, 0.9, 0.9], aspect='equal', anchor='SW')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xticks([])
ax.set_yticks([])
plt.hlines(list(range(11)), 0, 10, 'r', alpha=0.2)

# Create the button setup
ax1 = plt.axes([0.55, 0.8, 0.4, 0.15], anchor='SW')
B = Button(ax1, 'Drop a Needle', color='turquoise', hovercolor='teal')
s0 = '\nNeedles : {}'.format(NEEDLE_COUNT) + '\n\nIntersections : {}'.format(INTER_COUNT) + '\n'
s1 = '\n' + r' $\pi$ : {:.5f}'.format(np.pi) + '\n\n'
s2 = 'PI : {:.5f}'.format(PI) + '\n'
T = plt.figtext(0.55, 0.45, s0 + s1 + s2, fontdict={'family': 'Sans Serif', 'size': 16}, bbox=dict(facecolor='w'))
ax3 = plt.axes([0.78, 0.44, 0.17, 0.32], anchor='SW')
R = RadioButtons(ax3, ['x100', 'x10', 'x1'], 2, 'green')
ax4 = plt.axes([0.55, 0.065, 0.4, 0.33], anchor='SW')

# Radio button to set the multiplier for dropped needles
def radio_picker(label):
    global MUL
    d_count = {'x1': 1, 'x10': 10, 'x100': 100}
    MUL = d_count[label]

R.on_clicked(radio_picker)

# Button setup to drop needles into our designated area
def click(event):
    global PI, NEEDLE_COUNT, INTER_COUNT

    for i in range(MUL):
        N = Needle()
        N.draw_needle(ax)
        NEEDLE_COUNT += 1

        if N.is_intersecting():
            INTER_COUNT += 1
            PI = (2 * N.width) / (D * (INTER_COUNT / NEEDLE_COUNT))

        X.append(NEEDLE_COUNT)
        Y.append(PI)

    ax4.plot(X, Y, 'orange')
    ax4.hlines(np.pi, 0, NEEDLE_COUNT, color='brown', alpha=0.2)
    s3 = '\nNeedles : {}'.format(NEEDLE_COUNT) + '\n\nIntersections : {}'.format(INTER_COUNT) + '\n'
    s4 = 'PI : {}'.format(PI) + '\n'
    T.set_text(s3 + s1 + s4)
    fig.canvas.draw_idle()

B.on_clicked(click)

plt.show()
