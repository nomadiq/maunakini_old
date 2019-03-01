import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import matplotlib.cm
import math
from pathlib import Path


def plot2d(two_d_plane, base=1000000):

    cl = base * 1.1 ** np.arange(40)
    cmap = colormap.Blues_r
    fig = plt.figure(figsize=(8,8))
    a = fig.add_subplot(111)
    a.contour(two_d_plane.real, cl, cmap=cmap)
    plt.show()


def plot_2d_spectrum(spectrum, noise=1000000, levels=10, space=1.4, sign=None, save=None):
    cmap = matplotlib.cm.Reds_r  # contour map (colors to use for contours)
    contour_start = noise  # contour level start value
    contour_num = levels  # number of contour levels
    contour_factor = space  # scaling factor between contour levels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # plot the contours

    ax.contour(np.real(spectrum),
               cl,
               cmap=cmap,
               extent=(0, spectrum.shape[1] - 1, 0, spectrum.shape[0] - 1),
               linewidths=0.25,
               )

    if sign == 'PosNeg':  # plot the negative contours as well
        cl = -1 * contour_start * contour_factor ** np.arange(contour_num)
        cmap = matplotlib.cm.Greens
        ax.contour(np.real(spectrum), cl[::-1],
                   cmap=cmap,
                   extent=(0, spectrum.shape[1] - 1, 0, spectrum.shape[0] - 1),
                   linewidths=0.25,
                   )

    if save is not None:
        plt.savefig(save)

    plt.show()


def plot_2d_nuslist(nuslist):

    x = []
    y = []

    for samp in nuslist:
        x.append(samp[0])
        y.append(samp[1])

    plt.scatter(x, y)
