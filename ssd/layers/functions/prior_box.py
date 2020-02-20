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
        aspect_ratios = config['aspect_ratios']
        clip = config['clip']

        for k, f in enumerate(feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = image_size / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = min_sizes[k]/image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (max_sizes[k]/image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if clip:
            output.clamp_(max=1, min=0)
        return output
