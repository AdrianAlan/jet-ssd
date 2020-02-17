import math
import torch
import torch.nn as nn

from torch.autograd import Function


class BinaryLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.linear(input, self.weight)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinaryConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        self.bias.org = self.bias.data.clone()
        out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


def Binarize(tensor):
    t = tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5))
    return t.clamp_(0, 1).round().mul_(2).add_(-1)
