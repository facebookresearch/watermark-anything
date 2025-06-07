# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.transforms import rgb_to_yuv

class YUVLoss(nn.Module):
    """ MSE loss in YUV space """

    def __init__(self, 
        preprocess = lambda x: x,
    ):
        super(YUVLoss, self).__init__()
        self.preprocess = preprocess
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        loss = self.mse(
            rgb_to_yuv(self.preprocess(x)),
            rgb_to_yuv(self.preprocess(y))    
        )
        return loss