from __future__ import division
from math import sqrt as sqrt
import torch

from itertools import product as product
from torch.autograd import Function


class PriorBox(Function):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    @staticmethod
    def forward(ctx, config):

        mean = []
        image_size = config['min_dim']
        feature_maps = config['feature_maps']
        steps = config['steps']
        min_sizes = config['min_sizes']
        max_sizes = config['max_sizes']
        clip = config['clip']

        for k, f in enumerate(feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = image_size / steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k_min = min_sizes[k]/image_size
                mean += [cx, cy, s_k_min, s_k_min]

                s_k_max = max_sizes[k]/image_size
                mean += [cx, cy, s_k_max, s_k_max]

        output = torch.Tensor(mean).view(-1, 4)
        if clip:
            output.clamp_(max=1, min=0)
        return output
