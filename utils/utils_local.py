# Copyright Akhil Gurram 2021. Patent Pending. All rights reserved.

from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt, cm


def depth_visualize(rgb, depth_im, title_='',
                    cmap_name='turbo', fig_num=0, d_min=0.1, d_max=80.):
    cmap = cm.get_cmap(cmap_name)

    depth_im[depth_im < d_min] = d_min
    depth_im[depth_im > d_max] = d_max
    norm_depth = (depth_im - d_min) / (d_max - d_min)
    norm_depth = cmap(norm_depth)[:, :, :3]
    plt.figure(fig_num)
    plt.imshow(np.concatenate((rgb, norm_depth), 0))
    plt.title(title_)
    # plt.show()
    plt.pause(3)  # for 3 seconds
    return norm_depth


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# Print number of parameters in each model.
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
