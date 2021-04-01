# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 08:50:55 2020

@author: a84167753
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

relu_inplace = True
BatchNorm2d = BatchNorm2d_class = nn.BatchNorm2d

BN_MOMENTUM = 0.1
ALIGN_CORNERS = True


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(inChannel, outChannel, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, block, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):  # blocks -> blocktype
        super(HighResolutionModule, self).__init__()

        # check if the configuration is correct to develope a module
        self._check_branches(
            num_branches, block, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        # construct the branches by stacking blocks: returns nn.ModuleList with each element corresponding to one branch
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)

        # fuse the feature maps of different branches
        # (corresponds to the exchange unit -> see the paper)
        self.fuse_layers = self._make_fuse_layers()

        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        """checking wheter the parameters are correct to develope a module"""

        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        """Constructs one branch by stacking blocks (basic) sequentially
        output: nn.Sequential"""

        # if the input_channel dimension of a block and the output_channel dimension
        # of the block are not matching -> downsample/upsample residual
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        # stack layers sequentially to construct one branch
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Constructs branches of one module
        output: nn.ModuleList where each element is corresponding to one branch"""

        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks,
                                      num_channels))  # returns nn.Sequential -> blocks are serially stacked

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Fuses the feature maps of different branches in one module.
        Exchanges the information by fusing the feature maps.
        Explanation: (i) gets feature map information from (j)
        -> if j > i, spatial resolution of j branch is smaller:
            here in this method just 1x1 convolutional layer with channel dimension matching and then BN;
        -> if i==j None;
        -> if j < i -> j branch has higher spatial resolution:
            stacking 3x3 conv with stride 2
            f.e. if i-j > 1: j=0 branch has 100xHxW, i=2 branch has 200xH/4xW/4,
            then j->(3x3,stride=2,out_Channel=100,BN)->(3x3,stride=2,out_Channel=200,BN)
        Output: nn.ModuleList where each element is another nn.ModuleList with conv units for fusion"""

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        """Forwarding input x through one HR module.
        Input x: list of feature maps,
        where first element is the input feature map of the first branch,
        second element the feature map of the second branch and so on...
        feature maps from lower branches are bilinearly upsampled before elementwise sumation.
        Output: list of fused output feature maps of the one stage module"""

        # process input feature maps through the branches (blocks) to create output feature maps
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])  # x is processed through the branches where x is also list of inputs

        # fuse the output feature maps of the branches -> information exchange
        x_fuse = []

        # elementwise sum then relu to build xfuse
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class BasicBlock(nn.Module):
    """Basis block: represents 1 residual conv unit
    used in the second, third and fourth stages"""

    expansion = 1  # expansion/increase of the output channel dimension in the last conv layer of the unit

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.downsample = downsample  # downsample is a layer to adjust the channel dimension of the residual when necessary
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # when the channel dimension of x is changed (expansion) -> then adjust the channel dimension of residual
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block: represents 1 residual conv unit with bottleneck property
    used in the first stage"""

    expansion = 4  # for the bottleneck behaviour output_channel dimension of the last layer is increased by expansion_variable

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(out_channel * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(HRNet, self).__init__()

        global ALIGN_CORNERS
        ALIGN_CORNERS = config['HRNET_MODEL']['ALIGN_CORNERS']

        hrnet_config = config['HRNET_MODEL']['CONFIGURATION']

        # initial stem network
        stem_output_stride = hrnet_config['STEM_STRIDE']
        stem_input_channel = hrnet_config['STEM_INPUT_CHANNEL']
        self.stem_output_channel = hrnet_config['STEM_OUTPUT_CHANNEL']

        if stem_output_stride == 1:
            self.conv1 = conv3x3(stem_input_channel, self.stem_output_channel)

            self.bn1 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.conv2 = conv3x3(self.stem_output_channel, self.stem_output_channel)

            self.bn2 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=relu_inplace)
        elif stem_output_stride == 2:
            self.conv1 = conv3x3(stem_input_channel, self.stem_output_channel)

            self.bn1 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.conv2 = conv3x3(self.stem_output_channel, self.stem_output_channel, stride=2)

            self.bn2 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=relu_inplace)
        else:
            self.conv1 = conv3x3(stem_input_channel, self.stem_output_channel, stride=2)

            self.bn1 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.conv2 = conv3x3(self.stem_output_channel, self.stem_output_channel, stride=2)

            self.bn2 = BatchNorm2d(self.stem_output_channel, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=relu_inplace)

        # construct stage1 network
        self.stage1_config = hrnet_config['STAGE1']
        num_channels = self.stage1_config['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_config['BLOCK']]
        num_blocks = self.stage1_config['NUM_BLOCKS'][0]
        self.stage1 = self._make_layer(block, self.stem_output_channel, num_channels, num_blocks)
        stage1_out_channels = block.expansion * num_channels

        # construct stage2 network
        self.stage2_config = hrnet_config['STAGE2']
        num_channels = self.stage2_config['NUM_CHANNELS']
        block = blocks_dict[self.stage2_config['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition_stage_1_2 = self._make_transition_layer([stage1_out_channels], num_channels)
        self.stage2, stage2_out_channels = self._make_stage(self.stage2_config, num_channels)

        # construct stage3 network
        self.stage3_config = hrnet_config['STAGE3']
        num_channels = self.stage3_config['NUM_CHANNELS']
        block = blocks_dict[self.stage3_config['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition_stage_2_3 = self._make_transition_layer(stage2_out_channels, num_channels)
        self.stage3, stage3_out_channels = self._make_stage(self.stage3_config, num_channels)

        # construct stage4 network
        self.stage4_config = hrnet_config['STAGE4']
        num_channels = self.stage4_config['NUM_CHANNELS']
        block = blocks_dict[self.stage4_config['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition_stage_3_4 = self._make_transition_layer(stage3_out_channels, num_channels)
        self.stage4, stage4_out_channels = self._make_stage(self.stage4_config, num_channels)

        # initialize weights
        self.init_weights(config['HRNET_MODEL']['PRETRAINED'])

    def _make_stage(self, stage_config, num_inchannels, multi_scale_output=True):
        """constructs the stage:
            a stage is constructed by stacking equally designed HighResolutionModules
            the number of modules for each stage is defined in the config file.
            output: nn.Sequential of modules"""

        num_modules = stage_config['NUM_MODULES']
        num_branches = stage_config['NUM_BRANCHES']
        num_blocks = stage_config['NUM_BLOCKS']
        num_channels = stage_config['NUM_CHANNELS']
        block = blocks_dict[stage_config['BLOCK']]
        fuse_method = stage_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """constructs the transition from stage x to y (x<y):
            adds branches and adjusts channel dimensions.
            output: nn.ModuleList with each element corresponding to the
            transition of previous stage output to current stage input"""

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channel, out_channel, num_blocks, stride=1):
        """stacks blocks (basis/bottleneck block) sequentially.
        output: nn.Sequential"""

        # when the input_channel dimension and the output_channel dimension of the block is not consistent
        # downsample/upsamle the residual channel dimension
        downsample = None
        if stride != 1 or in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channel * block.expansion, momentum=BN_MOMENTUM),
            )

        # stack blocks sequentially
        layers = []
        layers.append(block(in_channel, out_channel, stride, downsample))
        in_channel = out_channel * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(in_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """forwarding the input (image) through the HRNet
        Input: stem_input_channel x H x W
        Output: Concatenation of feature maps from the final stage branches """

        # processing input through the stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # process x through stage1 -> 1 branch
        x = self.stage1(x)

        # output of stage1 is processed through the transition_layers
        # to generate input feature maps for the second stage
        x_list = []
        for i in range(self.stage2_config['NUM_BRANCHES']):
            if self.transition_stage_1_2[i] is not None:
                x_list.append(self.transition_stage_1_2[i](x))
            else:
                x_list.append(x)
        # process the input through the second stage
        y_list = self.stage2(x_list)

        # output of stage2 is processed through the transition_layers
        # to generate input feature maps for the third stage
        x_list = []
        for i in range(self.stage3_config['NUM_BRANCHES']):
            if self.transition_stage_2_3[i] is not None:
                if i < self.stage2_config['NUM_BRANCHES']:
                    x_list.append(self.transition_stage_2_3[i](y_list[i]))
                else:
                    x_list.append(self.transition_stage_2_3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # process the input through the third stage
        y_list = self.stage3(x_list)

        # output of stage3 is processed through the transition_layers
        # to generate input feature maps for the thourth/last stage
        x_list = []
        for i in range(self.stage4_config['NUM_BRANCHES']):
            if self.transition_stage_3_4[i] is not None:
                if i < self.stage3_config['NUM_BRANCHES']:
                    x_list.append(self.transition_stage_3_4[i](y_list[i]))
                else:
                    x_list.append(self.transition_stage_3_4[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # process the input through the last stage
        x = self.stage4(x_list)

        # Upsample all feature maps to the highest resolution
        # -> resolution of feature maps of the first branch x[0]
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1_0 = F.interpolate(x[1], size=(x0_h, x0_w),
                             mode='bilinear', align_corners=ALIGN_CORNERS)
        x2_0 = F.interpolate(x[2], size=(x0_h, x0_w),
                             mode='bilinear', align_corners=ALIGN_CORNERS)
        x3_0 = F.interpolate(x[3], size=(x0_h, x0_w),
                             mode='bilinear', align_corners=ALIGN_CORNERS)

        # concatenate all feature maps
        feats = torch.cat([x[0], x1_0, x2_0, x3_0], 1)  # concat in dimension 1 because:  batch x channelDim x H x W

        return feats

    def init_weights(self, pretrained_weights_path=''):
        """Initializes parameters or
        loads and updates the weights of the model with pretrained weights """

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained_weights_path):
            model_dict = self.state_dict()
            pretrained_dict = torch.load(pretrained_weights_path, map_location={'cuda:0': 'cpu'})

            model_pretrained_dict = {}
            for (k1, v1), (k2, v2) in zip(model_dict.items(), pretrained_dict.items()):
                if v1.shape == v2.shape:
                    model_pretrained_dict[k1] = v2
                else:
                    raise RuntimeError('Pretrained weights could not be correctly loaded!')
            model_dict.update(model_pretrained_dict)
            self.load_state_dict(model_dict)
            print('HRNet weights initialized with pretrained IMAGENET weights!')

        else:
            print('HRNET weights randomly initialized!')
