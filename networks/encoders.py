import numpy as np
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from networks import HRNet, resnet_multiimage_input


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
            feats_pyramid_0, size=[np.int(h / 2), np.int(w / 2)], mode='bilinear', align_corners=True)

        feats_pyramid_1 = self.pyramid_1(raw_hrnet_feats)
        feats_pyramid_2 = self.pyramid_2(raw_hrnet_feats)
        feats_pyramid_3 = self.pyramid_3(feats_pyramid_2)
        feats_pyramid_4 = self.pyramid_4(feats_pyramid_3)

        return [feats_pyramid_0, feats_pyramid_1, feats_pyramid_2, feats_pyramid_3,
                feats_pyramid_4], raw_hrnet_feats


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.features = []
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
