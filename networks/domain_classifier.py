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

from torch import nn
from torch.autograd import Function


# from the authors of https://github.com/Yangyangii/DANN-pytorch
class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.alpha = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha

        return output, None


class DomainClassifier(nn.Module):
    def __init__(self, in_channel=720, width=640, height=192, batch_size=8):
        super(DomainClassifier, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.down = nn.Conv2d(in_channel, 128, 5, 2, padding=2, bias=False)
        self.relu0 = nn.ReLU(True)
        self.fc1 = nn.Linear(int((128 * (width / 8) * (height / 8))), 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(100, 2)
        self.soft = nn.LogSoftmax(dim=1)

    def forward(self, feature, lambda_=1):
        feature = GradientReversalLayer.apply(feature, lambda_)

        feat = self.down(feature)
        feat = self.relu0(feat)
        feat = feat.view(-1, int(128 * (self.width / 8) * (self.height / 8)))
        feat = self.fc1(feat)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.fc2(feat)
        domain_output = self.soft(feat)

        return domain_output
