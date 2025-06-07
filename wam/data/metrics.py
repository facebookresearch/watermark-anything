# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import numpy as np

from .transforms import image_std

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.view(1, 3, 1, 1).to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    noise = torch.mean(delta**2, dim=(1,2,3))  # B
    psnr = peak - 10*torch.log10(noise)
    return psnr

def iou(preds, targets, threshold=0.0, label=1):
    """
    Return IoU for a specific label (0 or 1).
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
        label (int): The label to calculate IoU for (0 for background, 1 for foreground)
        threshold (float): Threshold to convert predictions to binary masks
    """
    preds = preds > threshold  # Bx1xHxW
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1,2,3))  # B
    union = (preds | targets).float().sum((1,2,3))  # B
    # avoid division by zero
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou

def accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Return accuracy
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
    """
    preds = preds > threshold  # b 1 h w
    targets = targets > 0.5
    correct = (preds == targets).float()  # b 1 h w
    accuracy = torch.mean(correct, dim=(1,2,3))  # b
    return accuracy

def bit_accuracy(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None, threshold: float = 0.0) -> torch.Tensor:
    """
    Computes the bit accuracy for each pixel, then averages over all pixels where the mask is not zero.
    This version supports multiple messages and corresponding masks.
    
    Args:
        preds (torch.Tensor): Predicted bits with shape [B, K, H, W]
        targets (torch.Tensor): Target bits with shape [B, Z, K]
        masks (torch.Tensor): Mask with shape [B, Z, H, W] (optional)
            Used to compute bit accuracy only on non-masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    if len(targets.shape) !=3:
        print(f"targets.shape: {targets.shape}")
        targets = targets.unsqueeze(1)
    preds = preds > threshold  # B, K, H, W
    targets = targets > 0.5  # B, Z, K
    correct = (preds.unsqueeze(1) == targets.unsqueeze(-1).unsqueeze(-1)).float()  # B, Z, K, H, W
    if masks is not None:
        masks = masks.unsqueeze(2)  # B, Z, 1, H, W to align with K dimension
        correct = correct * masks  # Apply masks
        bit_acc =  correct.sum() / (masks.sum() * correct.shape[2]) 
    # Optionally, handle NaNs if all pixels are masked
    # bit_acc = torch.nan_to_num(bit_acc, nan=0.0)
    return bit_acc

def bit_accuracy_inference(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    masks: torch.Tensor,
    method: str = 'hard',
    nb_repetitions: int = 1,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    assert preds.shape[1] % nb_repetitions == 0, preds.shape[1] % nb_repetitions
    a = preds.shape[1] // nb_repetitions
    for i in range(nb_repetitions-1):
        preds[:, :a, :, :] += preds[:, (1+i)*a:(i+2)*a, :, :]
    preds = preds[:, :a, :, :]
    targets = targets[:, :a]  # b k//nb_repetitions

    if method == 'hard':
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'semihard':
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'soft':
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  # b k
    preds = preds > threshold  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=(1))  # b
    return bit_acc

def msg_predict_inference(
    preds: torch.Tensor,
    masks: torch.Tensor,
    method: str = 'semihard',
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Computes the message by averaging over all pixels, then computes the bit accuracy.
    Closer to how the model is evaluated during inference.
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        masks (torch.Tensor): Mask with shape Bx1xHxW
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
        method (str): Method to compute bit accuracy. Options: 'hard', 'soft'
    """
    assert method in ['hard', 'semihard', 'soft'], f"Method {method} not supported"
    if method == 'hard':
        # convert every pixel prediction to binary, select based on masks, and average
        preds = preds > threshold  # b k h w
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'semihard':
        # select every pixel prediction based on masks, and average
        bsz, nbits, h, w = preds.size()
        masks = masks > 0.5  # b 1 h w
        masks = masks.expand_as(preds).bool()
        # masked select only works if all masks in the batch share the same number of 1s
        # not the case here, so we need to loop over the batch
        preds = [pred.masked_select(mask).view(nbits, -1) for mask, pred in zip(masks, preds)]  # b k n
        preds = [pred.mean(dim=-1, dtype=float) for pred in preds]  # b k
        preds = torch.stack(preds, dim=0)  # b k
    elif method == 'soft':
        # average every pixel prediction, use masks "softly" as weights for averaging
        bsz, nbits, h, w = preds.size()
        masks = masks.expand_as(preds)  # b k h w
        preds = torch.sum(preds * masks, dim=(2,3)) / torch.sum(masks, dim=(2,3))  # b k
    preds = preds > 0.5  # b k
    return preds

