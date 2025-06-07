# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Test with:
    python -m watermark_anything.augmentation.augmenter
"""

import os
from PIL import Image

import torch
from torch import nn
from torchvision.utils import save_image

from ..data.transforms import default_transform, unnormalize_img
from .masks import get_mask_embedder
from .geometric import Rotate, Resize, Crop, Perspective, HorizontalFlip, Identity, CropResizePad
from .valuemetric import JPEG, GaussianBlur, MedianFilter, Brightness, Contrast, Saturation, Hue


name2aug = {
    'rotate': Rotate,
    'resize': Resize,
    'crop': Crop,
    'perspective': Perspective,
    'hflip': HorizontalFlip,
    'identity': Identity,
    'jpeg': JPEG,
    'gaussian_blur': GaussianBlur,
    'median_filter': MedianFilter,
    'brightness': Brightness,
    'contrast': Contrast,
    'saturation': Saturation,
    'hue': Hue,
    'crop_resize_pad': CropResizePad,
}

class Augmenter(nn.Module):
    """
    Augments the watermarked image.
    """
    def __init__(
        self,
        masks: dict,
        augs: dict,
        augs_params: dict,
        **kwargs: dict
    ) -> None:
        super(Augmenter, self).__init__()

        # create mask embedder
        self.mask_embedder = get_mask_embedder(
            **masks  # contains, e.g., invert_proba 
        )

        # create augs
        self.post_augs, self.post_probs = self.parse_augmentations(
            augs = augs,
            augs_params = augs_params
        )

    def parse_augmentations(
        self, 
        augs: dict[str, float],
        augs_params: dict[str, dict[str, float]],
    ):
        """
        Parse the post augmentations into a list of augmentations.
        Args:
            augs: (dict) The augmentations to apply.
                e.g. {'identity': 4, 'resize': 1, 'crop': 1}
            augs_params: (dict) The parameters for each augmentation.
                e.g. {'resize': {'min_size': 0.7, 'max_size': 1.5}, 'crop': {'min_size': 0.7, 'max_size': 1.0}}
        """
        augmentations = []
        probs = []
        # parse each augmentation
        for aug_name in augs.keys():
            aug_prob = float(augs[aug_name])
            aug_params = augs_params[aug_name] if aug_name in augs_params else {}
            try:
                selected_aug = name2aug[aug_name](**aug_params)
            except KeyError:
                raise ValueError(f"Augmentation {aug_name} not found. Add it in name2aug.")
            augmentations.append(selected_aug)
            probs.append(aug_prob)
        # normalize probabilities 
        total_prob = sum(probs)
        probs = [prob / total_prob for prob in probs]  
        return augmentations, torch.tensor(probs)

    def post_augment(self, image, mask, do_resize=True):
        index = torch.multinomial(self.post_probs, 1).item()
        selected_aug = self.post_augs[index]
        if not do_resize:
            image, mask = selected_aug(image, mask)
        else:
            h, w = image.shape[-2:]
            image, mask = selected_aug(image, mask)
            if image.shape[-2:] != (h, w):
                image = nn.functional.interpolate(image, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
                mask = nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        return image, mask.int(), str(selected_aug)
    
    def forward(
        self, 
        imgs_w: torch.Tensor,
        imgs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            imgs_w: (torch.Tensor) Batched watermarked images with shape BxCxHxW
            masks: (torch.Tensor) Batched masks with shape Bx1xHxW
        Returns:
            imgs_aug: (torch.Tensor) The augmented watermarked images.
            mask_targets: (torch.Tensor) The mask targets, at ones where the watermark is present.
        """
        if self.training:
            # create mask targets 
            mask_targets = self.mask_embedder(imgs_w, masks=masks).to(imgs_w.device)
            # watermark masking
            imgs_aug = imgs_w * mask_targets + imgs * (1 - mask_targets)
            # image augmentations
            imgs_aug, mask_targets, selected_aug = self.post_augment(imgs_aug, mask_targets)
            return imgs_aug, mask_targets, selected_aug
        else:
            ### TOM CODE ###
            mask_targets = torch.ones_like(imgs_w)[:, 0:1, :, :]
            imgs_aug, mask_targets, selected_aug = self.post_augment(imgs_w, mask_targets)
            # imgs_aug = imgs_w
            return imgs_aug, mask_targets, selected_aug


    def __repr__(self) -> str:
        # print the augmentations and their probabilities
        augs = [aug.__class__.__name__ for aug in self.post_augs]
        return f"Augmenter(augs={augs}, probs={self.post_probs})"


if __name__ == "__main__":
    # Define the augmentations and their parameters
    augs = {
        'identity': 1, 
        'rotate': 1,
        'resize': 1,
        'crop': 1,
        # 'perspective': 1,
        'jpeg': 1,
        'gaussian_blur': 1,
        'median_filter': 1,
        'crop_resize_pad': 1,
    }
    augs_params = {
        'resize': {'min_size': 0.7, 'max_size': 1.5},
        'crop': {'min_size': 0.5, 'max_size': 0.7},
        'rotate': {'min_angle': -10, 'max_angle': 10},
        # 'perspective': {'distortion_scale': 0.5},
        'jpeg': {'min_quality': 40, 'max_quality': 80},
        'gaussian_blur': {'min_kernel_size': 3, 'max_kernel_size': 17},
        'median_filter': {'min_kernel_size': 3, 'max_kernel_size': 9},
        'crop_resize_pad': {'resize_min': 0.7, 'resize_max': 1.5, 'crop_min': 0.5, 'crop_max': 0.7},
    }

    masks = {
        "kind": "full",
    }

    # Create a batch of images
    imgs = [
        Image.open("assets/images/gauguin_256.png")
    ]
    imgs = torch.stack([default_transform(img) for img in imgs])
    imgs_w = imgs.clone()

    # Create an instance of the Augmenter class
    augmenter = Augmenter(
        masks=masks,
        augs=augs,
        augs_params=augs_params
    ).eval()
    print("Augmenter:", augmenter)

    # Apply the augmentations to the images and save
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    for ii in range(100): 
        imgs_aug, mask_targets, selected_aug = augmenter(imgs_w, imgs, masks)
        save_image(unnormalize_img(imgs_aug).clamp(0, 1), os.path.join(output_dir, f"imgs_aug_{ii}_{selected_aug}.png"), nrow=2)
        save_image(mask_targets, os.path.join(output_dir, f"mask_targets_{ii}_{selected_aug}.png"), nrow=2)
