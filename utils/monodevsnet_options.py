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

from __future__ import absolute_import, division, print_function

import argparse
import os
from datetime import date
from os.path import expanduser
from monodepth2 import MonodepthOptions

home = expanduser("~")
week_num = date.today().isocalendar()[1]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Additional arguments for MonoDEVSNet framework
class MonoDEVSOptions(MonodepthOptions):
    def __init__(self, base_path=os.path.join(os.path.abspath(os.getcwd()))):
        super(MonoDEVSOptions, self).__init__()
        # self.parser = argparse.ArgumentParser(description="Depth Domain Adaptation options")

        # PATHS
        self.parser.add_argument("--base_path",
                                 type=str,
                                 help="path to the monoDEVSNet framework",
                                 default=base_path)
        self.parser.add_argument("--real_data_path",
                                 type=str,
                                 help="path to real dataset",
                                 default=None)
        self.parser.add_argument("--syn_data_path",
                                 type=str,
                                 help="path to synthetic dataset",
                                 default=None)

        # Training options
        self.parser.add_argument("--use_dc",
                                 help="if set, uses domain classifier with gradient reversal layer at the output of"
                                      "encoder",
                                 action="store_true")
        self.parser.add_argument("--use_le",
                                 help="if set, uses loss equalization strategy to make sure supervision dominates self"
                                      "encoder",
                                 action="store_true")
        self.parser.add_argument("--use_ms",
                                 help="if set, uses loss equalization strategy to make sure supervision dominates self"
                                      "encoder",
                                 action="store_true")
        self.parser.add_argument("--use_segm",
                                 help="to use segmentation block/loss or not",
                                 action="store_true")

        #   Model options
        self.parser.add_argument('--config',
                                 help='configuration of the hrnet encoder',
                                 type=str,
                                 default='hrnet_w48_vk2')
        self.parser.add_argument("--models_fcn_name",
                                 nargs="+",
                                 type=str,
                                 help="mention the network model names of encoder, decoders, domain classifier",
                                 default=[])

        #   Dataset options
        self.parser.add_argument("--train_with",
                                 type=str,
                                 help="training with real or syn or both",
                                 default="both",
                                 choices=["real", "syn", "both"])
        self.parser.add_argument("--real_dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--syn_dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="vk_2.0",
                                 choices=["vk_1.0", "vk_2.0"])
        self.parser.add_argument("--total_number_of_images_for_training",
                                 help="Total number of images used for training from synthetic and real images",
                                 type=int,
                                 default=-1)
        self.parser.add_argument("--depth_init",
                                 help="provide starting depth values",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--n_class",
                                 help="Number of semantic segmentation classes",
                                 type=int,
                                 default=17)

        #   Loss function options
        self.parser.add_argument("--loss_fcn",
                                 help="use multiple loss functions: L1, L2, Huber, BerHu, self-supervised, ...",
                                 type=str,
                                 default='')
        self.parser.add_argument("--syn_loss_fcn",
                                 help="use multiple loss functions: L1, L2, Huber, BerHu, self-supervised, ...",
                                 type=str,
                                 default='l1')
        self.parser.add_argument("--real_loss_fcn",
                                 help="use multiple loss functions: L1, L2, Huber, BerHu, self-supervised, ...",
                                 type=str,
                                 default='self')

        #   constant parameters/options
        self.parser.add_argument("--syn_scaling_factor", help=" mention trainer names", type=float,
                                 default=3.0)
        self.parser.add_argument("--self_scaling_factor", help=" weighting parameter on self-supervised", type=float,
                                 default=100.)

        # SYSTEM options
        self.parser.add_argument("--cuda_idx",
                                 help="if set disables CUDA",
                                 type=int,
                                 default=0)

        # Augmentation options
        self.parser.add_argument('--do_flip', dest='do_flip', const=False,
                                 default=True, type=str2bool, nargs='?',
                                 help='Flip the images during training randomly.')
        self.parser.add_argument('--do_color_aug', dest='do_color_aug', const=False,
                                 default=True, type=str2bool, nargs='?',
                                 help='Do color augmentation during training randomly.')

        self.parser.add_argument("--version",
                                 help="version name",
                                 type=str,
                                 default='markI')
        self.options = []

    def parse(self):
        self.options = self.parser.parse_args()

        changed_names = convert_list2dict(self.options.models_fcn_name)
        default_class = {"encoder": "HRNet", "depth_decoder": "DepthDecoder",
                         "pose_encoder": "ResnetEncoder", "pose": "PoseDecoder",
                         "domain_classifier": "DomainClassifier"}
        for k, v in changed_names.items():
            default_class[k] = v

        self.options.models_fcn_name = default_class.copy()

        self.options.models_to_load = ["encoder", "depth_decoder", "pose_encoder", "pose"]
        return self.options


def convert_list2dict(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct
