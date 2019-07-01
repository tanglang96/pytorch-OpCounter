import argparse

import torch
import torch.nn as nn

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.pad_h = 0
        self.pad_w = 0
        self.pad_list = []

    def forward(self, x):
        # print(x.shape)
        # global padding_dict
        # ih, iw = x.size()[-2:]
        # kh, kw = self.weight.size()[-2:]
        # sh, sw = self.stride
        # oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        # # oh, ow = ih // sh + 1, iw // sw + 1
        # pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        # pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        # print('(%d,%d)' % (pad_h, pad_w))
        # padding_dict.append((pad_h, pad_w))
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, self.pad_list)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)