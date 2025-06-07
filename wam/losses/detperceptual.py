# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# adapted from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpips import LPIPS

from .perceptual import PerceptualLoss
from ..modules.discriminator import NLayerDiscriminator
from ..data.transforms import imnet_to_lpips

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
        balanced = True, total_norm = 0.0, 
        disc_weight = 1.0, percep_weight = 1.0, detect_weight = 1.0, decode_weight = 0.0,
        disc_start = 0, disc_num_layers = 3, disc_in_channels = 3, disc_loss = "hinge", use_actnorm = False, 
        percep_loss = "lpips"
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        self.balanced = balanced
        self.total_norm = total_norm

        self.percep_weight = percep_weight
        self.detect_weight = detect_weight
        self.disc_weight = disc_weight
        self.decode_weight = decode_weight

        self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,n_layers=disc_num_layers,use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else nn.BCEWithLogitsLoss()

        self.detection_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.decoding_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def calculate_adaptive_weights(
        self, 
        losses, 
        weights, 
        last_layer, 
        total_norm=0, 
        choose_norm_idx=2,
        eps=1e-12
    ) -> list:
        # calculate gradients for each loss
        grads = []
        for loss in losses:
            # allows for the computation of gradients w.r.t. intermediate layers if possible
            try:
                grads.append(torch.autograd.grad(loss, last_layer, retain_graph=True)[0])
            except:
                grads.append(torch.zeros_like(last_layer))
        grad_norms = [torch.norm(grad) for grad in grads]

        # calculate base weights
        total_weight = sum(weights)
        ratios = [w / total_weight for w in weights]

        # choose total_norm to be the norm of the biggest weight
        assert choose_norm_idx or total_norm > 0, "Either choose_norm_idx or total_norm should be provided"
        if total_norm <= 0:  # if not provided, use the norm of the biggest weight
            # max_idx = ratios.index(max(ratios))
            max_idx = choose_norm_idx
            total_norm = grad_norms[max_idx]

        # calculate adaptive weights
        scales = [r * total_norm / (eps + norm) for r, norm in zip(ratios, grad_norms)]
        return scales

    def forward(self, 
        inputs: torch.Tensor, reconstructions: torch.Tensor, 
        masks: torch.Tensor, msgs: torch.Tensor, preds: torch.Tensor,
        optimizer_idx: int, global_step: int, 
        last_layer=None, cond=None, msgs2 = None
    ):
        
        if optimizer_idx == 0:  # embedder update
            weights = [self.percep_weight, self.disc_weight, self.detect_weight]
            losses = []
            # perceptual loss
            losses.append(self.perceptual_loss(
                imgs = imnet_to_lpips(inputs.contiguous()),
                imgs_w = imnet_to_lpips(reconstructions.contiguous()),
            ).mean())  
            # discriminator loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
            losses.append( - logits_fake.mean())
            # detection loss
            detection_loss = self.detection_loss(preds[:, 0:1, :, :].contiguous(), masks.max(1).values.unsqueeze(1).contiguous().float()).mean()
            losses.append(detection_loss)
            # decoding loss
            losses_decoding = []
            if len(msgs.shape) > 1 and self.decode_weight > 0:
                # flatten and select pixels where mask is =1
                msg_preds = preds[:, 1:, :, :]  # b nbits h w
                for num_msg in range(msgs.shape[1]):
                    msg_targs = msgs[:, num_msg, :]
                    msg_targs = msg_targs.unsqueeze(-1).unsqueeze(-1).expand_as(msg_preds)  # b nbits h w
                    mask = masks[:, num_msg].unsqueeze(1).expand_as(msg_preds).bool()
                    msg_preds_ = msg_preds.masked_select(mask)
                    msg_targs = msg_targs.masked_select(mask)
                    # non empty mask
                    if len(msg_preds_)>0:
                        losses_decoding.append(self.decoding_loss(msg_preds_.contiguous(), msg_targs.contiguous().float())) 
                if len(losses_decoding) > 0:
                    combined_loss = torch.cat(losses_decoding, dim=0)
                    average_loss = combined_loss.mean()
                    losses.append(average_loss)
                    weights.append(self.decode_weight)
            # calculate adaptive weights
            if last_layer is not None and self.balanced:
                scales = self.calculate_adaptive_weights(
                    losses = losses,
                    weights = weights,
                    last_layer = last_layer,
                    total_norm = self.total_norm,
                )
            else:
                scales = weights
            total_loss = sum(scales[i] * losses[i] for i in range(len(losses)))
            # log
            log = {
                "total_loss": total_loss.clone().detach().mean(),
                "percep_loss": losses[0].clone().detach().mean(),
                "percep_scale": scales[0],
                "disc_loss": losses[1].clone().detach().mean(),
                "disc_scale": scales[1],
                "detect_loss": losses[2].clone().detach().mean(),
                "detect_scale": scales[2],
                "decode_loss": losses[3].clone().detach().mean() if len(losses) > 3 else -1,#0.0,
                "decode_scale": scales[3] if len(losses) > 3 else -1,#0.0,
            }
            return total_loss, log

        if optimizer_idx == 1:  # discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "disc_factor": disc_factor,
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log
    
    def to(self, device, *args, **kwargs):
        """
        Override for custom perceptual loss to device.
        """
        super().to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        return self