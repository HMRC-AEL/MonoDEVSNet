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

import numpy as np
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from networks import HRNet, densenet121, densenet169, densenet161, densenet201


def _conv_unit(in_channels, out_channels, kernel_size, stride, padding=1):
    conv_unit = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU())

    return conv_unit


class HRNetPyramidEncoder(nn.Module):

    def __init__(self, config_hrnet):
        super(HRNetPyramidEncoder, self).__init__()
        self.hrnet = HRNet(config_hrnet)

        hrnet_final_channels = sum(config_hrnet['HRNET_MODEL']['CONFIGURATION']['STAGE4']['NUM_CHANNELS'])
        pyramid_0_ch = config_hrnet['HRNET_MODEL']['PYRAMID_CHANNELS'][0]
        pyramid_1_ch = config_hrnet['HRNET_MODEL']['PYRAMID_CHANNELS'][1]
        pyramid_2_ch = config_hrnet['HRNET_MODEL']['PYRAMID_CHANNELS'][2]
        pyramid_3_ch = config_hrnet['HRNET_MODEL']['PYRAMID_CHANNELS'][3]
        pyramid_4_ch = config_hrnet['HRNET_MODEL']['PYRAMID_CHANNELS'][4]

        self.pyramid_0 = _conv_unit(hrnet_final_channels, pyramid_0_ch, kernel_size=3, stride=1, padding=1)
        self.pyramid_1 = _conv_unit(hrnet_final_channels, pyramid_1_ch, kernel_size=3, stride=1, padding=1)
        self.pyramid_2 = _conv_unit(hrnet_final_channels, pyramid_2_ch, kernel_size=3, stride=2, padding=1)
        self.pyramid_3 = _conv_unit(pyramid_2_ch, pyramid_3_ch, kernel_size=3, stride=2, padding=1)
        self.pyramid_4 = _conv_unit(pyramid_3_ch, pyramid_4_ch, kernel_size=3, stride=2, padding=1)

        self.num_ch_enc = np.array([pyramid_0_ch, pyramid_1_ch, pyramid_2_ch, pyramid_3_ch, pyramid_4_ch])

    def forward(self, input_image):
        batch, ch, h, w = input_image.size()
        x = (input_image - 0.45) / 0.225

        # generate default feature maps by processing x through Hrnet
        raw_hrnet_feats = self.hrnet(x)

        # pyramid of feature maps
        feats_pyramid_0 = self.pyramid_0(raw_hrnet_feats)
        feats_pyramid_0 = F.interpolate(
            feats_pyramid_0, size=[np.int32(h / 2), np.int32(w / 2)], mode='bilinear', align_corners=True)

        feats_pyramid_1 = self.pyramid_1(raw_hrnet_feats)
        feats_pyramid_2 = self.pyramid_2(raw_hrnet_feats)
        feats_pyramid_3 = self.pyramid_3(feats_pyramid_2)
        feats_pyramid_4 = self.pyramid_4(feats_pyramid_3)

        return [feats_pyramid_0, feats_pyramid_1, feats_pyramid_2, feats_pyramid_3,
                feats_pyramid_4], raw_hrnet_feats


class DensenetPyramidEncoder(nn.Module):

    def __init__(self, densnet_version=121, pretrained_weights=True):
        super(DensenetPyramidEncoder, self).__init__()
        if 121 == densnet_version:
            self.densenet = densenet121(pretrained=pretrained_weights)
            layers = [6, 12, 24, 16]
            num_init_channels = 64
            channel_per_layer = 32
        elif 169 == densnet_version:
            self.densenet = densenet169(pretrained=pretrained_weights)
            layers = [6, 12, 32, 32]
            num_init_channels = 64
            channel_per_layer = 32
        elif 161 == densnet_version:
            self.densenet = densenet161(pretrained=pretrained_weights)
            layers = [6, 12, 36, 24]
            num_init_channels = 96
            channel_per_layer = 48
        elif 201 == densnet_version:
            self.densenet = densenet201(pretrained=pretrained_weights)
            layers = [6, 12, 48, 32]
            num_init_channels = 64
            channel_per_layer = 32

        pyramid_in_channels = [num_init_channels]
        for i in range(0, len(layers)):
            if i == 0:
                pyramid_in_channels.append(layers[i] * channel_per_layer + pyramid_in_channels[i])
            else:
                pyramid_in_channels.append(int(layers[i] * channel_per_layer + pyramid_in_channels[i] / 2))

        self.num_ch_enc = np.array(pyramid_in_channels)

    def forward(self, input_image):
        batch, ch, h, w = input_image.size()
        x = (input_image - 0.45) / 0.225

        # generate default feature maps by processing x through Hrnet
        densenet_feats = self.densenet(x)

        # pyramid of feature maps
        feats_pyramid_0 = densenet_feats[0]
        feats_pyramid_0 = F.interpolate(
            feats_pyramid_0, size=[np.int(h / 2), np.int(w / 2)], mode='bilinear', align_corners=True)

        feats_pyramid_1 = densenet_feats[1]
        feats_pyramid_2 = densenet_feats[2]
        feats_pyramid_3 = densenet_feats[3]
        feats_pyramid_4 = densenet_feats[4]
        raw_densenet_feats = densenet_feats[4]

        return [feats_pyramid_0, feats_pyramid_1, feats_pyramid_2, feats_pyramid_3,
                feats_pyramid_4], raw_densenet_feats
