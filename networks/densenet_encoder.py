import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch import Tensor
from typing import Any, List, Tuple

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.bottleneck = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features

        self.features = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # for i, num_layers in enumerate(block_config):
        #     block = _DenseBlock(
        #         num_layers=num_layers,
        #         num_input_features=num_features,
        #         bn_size=bn_size,
        #         growth_rate=growth_rate,
        #         drop_rate=drop_rate,
        #         memory_efficient=memory_efficient
        #     )
        #     self.features.add_module('denseblock%d' % (i + 1), block)
        #     num_features = num_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans = _Transition(num_input_features=num_features,
        #                             num_output_features=num_features // 2)
        #         self.features.add_module('transition%d' % (i + 1), trans)
        #         num_features = num_features // 2

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.num_ch_enc = np.array([64, 512, 512, 1024, 1024])

    def forward(self, x: Tensor) -> Tensor:
        features_list = []
        features = self.bottleneck(x)
        features_list.append(features)
        for idx, module in enumerate(self.features):
            features = module(features)
            if idx % 2 == 0:
                features_list.append(features)

        return features_list


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    pretrained_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(pretrained_dict.keys()):
        if ('conv0' in key) or ('norm0' in key):
            new_key = 'bottleneck' + key[8:]
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]
            key = new_key
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]
    for key in list(pretrained_dict.keys()):
        if ('classifier' in key) or ('norm5' in key):
            del pretrained_dict[key]

    # model_dict = model.state_dict()
    # #pretrained_dict = load_state_dict_from_url(model_url, progress=progress)
    #
    # model_pretrained_dict = {}
    # for (k1, v1), (k2, v2) in zip(model_dict.items(), pretrained_dict.items()):
    #     if 'track' in k1:
    #         continue
    #     print(k1)
    #     print(k2)
    #     print('-------------------')
    #     if v1.shape == v2.shape:
    #         model_pretrained_dict[k1] = v2
    #     else:
    #         raise RuntimeError('Pretrained weights could not be correctly loaded!')
    # model_dict.update(model_pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Densenet weights initialized with pretrained IMAGENET weights!')

    model.load_state_dict(pretrained_dict)


def _densenet(
        arch: str,
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)


def _conv_unit(in_channels, out_channels, kernel_size, stride, padding=1):
    conv_unit = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU())
    return conv_unit


class DensenetPyramidEncoderTest(nn.Module):

    def __init__(self, densnet_version=121, pretrained_weights=True):
        super(DensenetPyramidEncoderTest, self).__init__()
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
        else:
            num_init_channels, layers, channel_per_layer = 0, [], 0
            raise RuntimeError('Choose densenet version within your limits')

        pyramid_channels = [64, 512, 512, 1024, 1024]
        pyramid_in_channels = [num_init_channels]
        for i in range(0, len(layers)):
            if i == 0:
                pyramid_in_channels.append(layers[i] * channel_per_layer + pyramid_in_channels[i])
            else:
                pyramid_in_channels.append(int(layers[i] * channel_per_layer + pyramid_in_channels[i] / 2))

        self.pyramid_0 = _conv_unit(pyramid_in_channels[0], pyramid_channels[0], kernel_size=3, stride=1, padding=1)
        self.pyramid_1 = _conv_unit(pyramid_in_channels[1], pyramid_channels[1], kernel_size=3, stride=1, padding=1)
        self.pyramid_2 = _conv_unit(pyramid_in_channels[2], pyramid_channels[2], kernel_size=3, stride=1, padding=1)
        self.pyramid_3 = _conv_unit(pyramid_in_channels[3], pyramid_channels[3], kernel_size=3, stride=1, padding=1)
        self.pyramid_4 = _conv_unit(pyramid_in_channels[4], pyramid_channels[4], kernel_size=3, stride=1, padding=1)

        self.num_ch_enc = np.array(pyramid_channels)

    def forward(self, input_image):
        batch, ch, h, w = input_image.size()
        x = (input_image - 0.45) / 0.225

        # generate default feature maps by processing x through Hrnet
        densenet_feats = self.densenet(x)

        # pyramid of feature maps
        feats_pyramid_0 = self.pyramid_0(densenet_feats[0])
        feats_pyramid_0 = F.interpolate(
            feats_pyramid_0, size=[np.int(h / 2), np.int(w / 2)], mode='bilinear', align_corners=True)

        feats_pyramid_1 = self.pyramid_1(densenet_feats[1])
        feats_pyramid_2 = self.pyramid_2(densenet_feats[2])
        feats_pyramid_3 = self.pyramid_3(densenet_feats[3])
        feats_pyramid_4 = self.pyramid_4(densenet_feats[4])
        raw_densenet_feats = densenet_feats[4]

        return [feats_pyramid_0, feats_pyramid_1, feats_pyramid_2, feats_pyramid_3,
                feats_pyramid_4], raw_densenet_feats
