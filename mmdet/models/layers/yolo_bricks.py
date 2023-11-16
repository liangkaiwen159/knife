# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, MaxPool2d, build_norm_layer)
from mmdet.models.layers.csp_layer import \
    DarknetBottleneck as MMDET_DarknetBottleneck
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class ELANBlock(BaseModule):
    """Efficient layer aggregation networks for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        middle_ratio (float): The scaling ratio of the middle layer
            based on the in_channels.
        block_ratio (float): The scaling ratio of the block layer
            based on the in_channels.
        num_blocks (int): The number of blocks in the main branch.
            Defaults to 2.
        num_convs_in_block (int): The number of convs pre block.
            Defaults to 1.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 middle_ratio: float,
                 block_ratio: float,
                 num_blocks: int = 2,
                 num_convs_in_block: int = 1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_blocks >= 1
        assert num_convs_in_block >= 1

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * block_ratio)
        final_conv_in_channels = int(num_blocks * block_channels) + 2 * middle_channels

        self.main_conv = ConvModule(in_channels,
                                    middle_channels,
                                    1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)

        self.short_conv = ConvModule(in_channels,
                                     middle_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_convs_in_block == 1:
                internal_block = ConvModule(middle_channels,
                                            block_channels,
                                            3,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg)
            else:
                internal_block = []
                for _ in range(num_convs_in_block):
                    internal_block.append(
                        ConvModule(middle_channels,
                                   block_channels,
                                   3,
                                   padding=1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg))
                    middle_channels = block_channels
                internal_block = nn.Sequential(*internal_block)

            middle_channels = block_channels
            self.blocks.append(internal_block)

        self.final_conv = ConvModule(final_conv_in_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.final_conv(x_final)


@MODELS.register_module()
class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = MODELS.build(act_cfg)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=groups,
                                        bias=False,
                                        norm_cfg=norm_cfg,
                                        act_cfg=None)
            self.rbr_1x1 = ConvModule(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=padding_11,
                                      groups=groups,
                                      bias=False,
                                      norm_cfg=norm_cfg,
                                      act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class BottleRep(nn.Module):
    """Bottle Rep Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        adaptive_weight (bool): Add adaptive_weight when forward calculate.
            Defaults False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_cfg=dict(type='RepVGGBlock'),
                 adaptive_weight: bool = False):
        super().__init__()
        conv1_cfg = block_cfg.copy()
        conv2_cfg = block_cfg.copy()

        conv1_cfg.update(dict(in_channels=in_channels, out_channels=out_channels))
        conv2_cfg.update(dict(in_channels=out_channels, out_channels=out_channels))

        self.conv1 = MODELS.build(conv1_cfg)
        self.conv2 = MODELS.build(conv2_cfg)

        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if adaptive_weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class MaxPoolAndStrideConvBlock(BaseModule):
    """Max pooling and stride conv layer for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        maxpool_kernel_sizes (int): kernel sizes of pooling layers.
            Defaults to 2.
        use_in_channels_of_middle (bool): Whether to calculate middle channels
            based on in_channels. Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 maxpool_kernel_sizes: int = 2,
                 use_in_channels_of_middle: bool = False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        middle_channels = in_channels if use_in_channels_of_middle \
            else out_channels // 2

        self.maxpool_branches = nn.Sequential(
            MaxPool2d(kernel_size=maxpool_kernel_sizes, stride=maxpool_kernel_sizes),
            ConvModule(in_channels, out_channels // 2, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.stride_conv_branches = nn.Sequential(
            ConvModule(in_channels, middle_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(middle_channels,
                       out_channels // 2,
                       3,
                       stride=2,
                       padding=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        maxpool_out = self.maxpool_branches(x)
        stride_conv_out = self.stride_conv_branches(x)
        return torch.cat(
            [stride_conv_out, maxpool_out],
            dim=1,
        )


@MODELS.register_module()
class SPPFCSPBlock(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer with CSP for
     YOLOv7

     Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         expand_ratio (float): Expand ratio of SPPCSPBlock.
            Defaults to 0.5.
         kernel_sizes (int, tuple[int]): Sequential or number of kernel
             sizes of pooling layers. Defaults to 5.
         is_tiny_version (bool): Is tiny version of SPPFCSPBlock. If True,
            it means it is a yolov7 tiny model. Defaults to False.
         conv_cfg (dict): Config dict for convolution layer. Defaults to None.
             which means using conv2d. Defaults to None.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.03, eps=0.001).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         init_cfg (dict or list[dict], optional): Initialization config dict.
             Defaults to None.
     """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 is_tiny_version: bool = False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.is_tiny_version = is_tiny_version

        mid_channels = int(2 * out_channels * expand_ratio)

        if is_tiny_version:
            self.main_layers = ConvModule(in_channels,
                                          mid_channels,
                                          1,
                                          conv_cfg=conv_cfg,
                                          norm_cfg=norm_cfg,
                                          act_cfg=act_cfg)
        else:
            self.main_layers = nn.Sequential(
                ConvModule(in_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(mid_channels,
                           mid_channels,
                           3,
                           padding=1,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg),
                ConvModule(mid_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            )

        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        else:
            self.poolings = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])

        if is_tiny_version:
            self.fuse_layers = ConvModule(4 * mid_channels,
                                          mid_channels,
                                          1,
                                          conv_cfg=conv_cfg,
                                          norm_cfg=norm_cfg,
                                          act_cfg=act_cfg)
        else:
            self.fuse_layers = nn.Sequential(
                ConvModule(4 * mid_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(mid_channels,
                           mid_channels,
                           3,
                           padding=1,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        self.short_layer = ConvModule(in_channels,
                                      mid_channels,
                                      1,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)

        self.final_conv = ConvModule(2 * mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

    def forward(self, x) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x1 = self.main_layers(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x1)
            y2 = self.poolings(y1)
            concat_list = [x1] + [y1, y2, self.poolings(y2)]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))
        else:
            concat_list = [x1] + [m(x1) for m in self.poolings]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))

        x2 = self.short_layer(x)
        return self.final_conv(torch.cat((x1, x2), dim=1))


class ImplicitA(nn.Module):
    """Implicit add layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 0.
        std (float): Std value of implicit module. Defaults to 0.02
    """

    def __init__(self, in_channels: int, mean: float = 0., std: float = .02):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplier layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 1.
        std (float): Std value of implicit module. Defaults to 0.02.
    """

    def __init__(self, in_channels: int, mean: float = 1., std: float = .02):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit * x
