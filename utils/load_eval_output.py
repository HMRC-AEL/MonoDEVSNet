# Author: Akhil Gurram
# Build on top of the monodepth2
# (Automatically pulled from git repo, monodepth2 source code is not included in this repository)
# This is the training script of the MonoDEVSNet framework.
# MonoDEVSNet: Monocular Depth Estimation through Virtual-world Supervision and Real-world SfM Self-Supervision
# https://arxiv.org/abs/2103.12209

# MIT License
#
# Copyright (c) 2021 Huawei Technologies Duesseldorf GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import torch
import argparse
from tqdm import tqdm

from utils import depth_visualize

parser = argparse.ArgumentParser(description="MonoDEVSNet test options")
parser.add_argument("--file_path",
                    type=str,
                    help="path to the MonoDEVSNet's precompute depth maps",
                    default='')
opts = parser.parse_args()


# rgbs: num_images x height x width x channels
# depth: num_images x 1 x height x width
def load_eval_output(pth_file_path):
    data = torch.load(pth_file_path)
    rgbs = data['rgbs']
    pred_depths = data['pred_depths']
    for i in tqdm(range(len(rgbs))):
        rgb = rgbs[i]
        pred_depth = pred_depths[i]
        depth_visualize(rgb, pred_depth, title_=str(i))


if __name__ == '__main__':
    try:
        load_eval_output(opts.file_path)
    except Exception as e:
        print(e)
