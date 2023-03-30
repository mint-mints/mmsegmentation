import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class PPLiteSegHead(BaseDecodeHead):
    """
    The head of PPLiteSeg.
    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module. (c6,c7,c8)
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self,
                 # backbone_out_chs,    => in_channels
                 arm_out_chs=[32, 64, 128],
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 seg_head_inter_chs=[64, 64, 64],
                 arm_type='UAFM_SpAtten',
                 resize_mode='bilinear',
                 is_training=True,
                 **kwargs):

        # self.training = is_training
        super(PPLiteSegHead, self).__init__(input_transform='multiple_select', **kwargs)
        # super().__init__()

        self.cm = PPContextModule(self.in_channels[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        arm_class = eval(arm_type)

        self.arm_list = nn.ModuleList()  # [..., arm8, arm16, arm32]
        for i in range(len(self.in_channels)):
            low_chs = self.in_channels[i]
            high_ch = cm_out_ch if i == len(
                self.in_channels) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

        # self.seg_heads = nn.ModuleList()  # [..., head_16, head32]
        # self.seg_heads = SegHead(arm_out_chs[0], seg_head_inter_chs[0], self.num_classes)
        # for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
        #     self.seg_heads.append(SegHead(in_ch, mid_ch, self.num_classes))

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and feat_list are the same.
        """

        high_feat = self.cm(in_feat_list[-1])
        feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            feat_list.insert(0, high_feat)

        # if self.training:
        #     logit_list = []
        #
        #     for x, seg_head in zip(feat_list, self.seg_heads):
        #         x = seg_head(x)
        #         logit_list.append(x)
        #
        #     logit_list = [
        #         F.interpolate(
        #             x, x_hw, mode='bilinear', align_corners=False)
        #         for x in logit_list
        #     ]
        # else:
        #     x = self.seg_heads[0](feat_list[0])
        #     x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
        #     logit_list = [x]

        # return logit_list

        output = self.cls_seg(feat_list[0])
        return output


    # def cls_seg(self, feat):
    #     """Classify each pixel."""
    #     if self.dropout is not None:
    #         feat = self.dropout(feat)
    #     output = self.conv_seg(feat)
    #     return output


class PPContextModule(nn.Module):
    """
    Simple Context module.
    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_planes=inter_channels,
            out_planes=out_channels,
            kernel=3)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_planes=in_channels, out_planes=out_channels, kernel=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan,
            mid_chan,
            kernel=3,
            stride=1
            )
        self.conv_out = nn.Conv2d(
            mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel=ksize)
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel=3)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel=3),
            ConvBN(
                2, 1, kernel=3))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        # print(atten.shape)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ConvBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # print("mean_value: ", mean_value)
    # print("max_value: ", max_value)

    if use_concat:
        res = torch.cat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res

def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.cat(res, dim=1)