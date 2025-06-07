# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from lpips import LPIPS

from .ssim import SSIM, MSSSIM
from .yuv import YUVLoss

class NoneLoss(nn.Module):
    def forward(self, x, y):
        return torch.zeros(1, requires_grad=True)

class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        percep_loss: str
    ):
        super(PerceptualLoss, self).__init__()
        self.losses = {
            "lpips": LPIPS(net="vgg").eval(),
            "mse": nn.MSELoss(),
            "yuv": YUVLoss(),
            "none": NoneLoss(),
            "ssim": SSIM(),
            "msssim": MSSSIM(),
        }
        self.percep_loss = percep_loss
        self.perceptual_loss = self.create_perceptual_loss(percep_loss)

    def create_perceptual_loss(
        self, 
        percep_loss: str
    ):
        """
        Create a perceptual loss function from a string.
        Args:
            percep_loss: (str) The perceptual loss string.
                Example: "lpips", "lpips+mse", "lpips+0.1_mse", ...
        """
        parts = percep_loss.split('+')
        if len(parts) == 1 and parts[0] in self.losses:
            return self.losses[parts[0]]
        
        def combined_loss(x, y):
            total_loss = 0
            for part in parts:
                if '_' in part:  # Check if the format is 'weight_loss'
                    weight, loss_key = part.split('_')
                else:
                    weight, loss_key = 1, part
                weight = float(weight)
                if loss_key in self.losses:
                    total_loss += weight * self.losses[loss_key](x, y).mean()
                else:
                    raise ValueError(f"Loss type {loss_key} not supported.")
            return total_loss
        
        return combined_loss

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        return self.perceptual_loss(imgs, imgs_w)

    def to(self, device, *args, **kwargs):
        """
        Override the to method to move only some of the perceptual loss functions to the device.
        + the losses are not moved by default since they are in a dict.
        """
        super().to(device)
        activated = []
        for loss in self.losses.keys():
            if loss in self.percep_loss:
                activated.append(loss)
        for loss in activated:
            self.losses[loss] = self.losses[loss].to(device)
        return self

    def __repr__(self):
        return f"PerceptualLoss(percep_loss={self.percep_loss})"