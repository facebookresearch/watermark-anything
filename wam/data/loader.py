# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import functools
import numpy as np
from pycocotools import mask as maskUtils
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from torchvision.datasets import CocoDetection
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision.transforms import Compose, ToTensor, Normalize

from ..utils.dist import is_dist_avail_and_initialized
from .transforms import default_transform


@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img), 0
        return img, 0

    def __len__(self):
        return len(self.samples)


def get_dataloader(
    data_dir: str, 
    transform: callable = default_transform,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8
) -> DataLoader:
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle) 
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=sampler, num_workers=num_workers, 
                                pin_memory=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers, 
                                pin_memory=True, drop_last=True)
    return dataloader


import numpy as np
import torch
from pycocotools import mask as maskUtils

class CocoImageIDWrapper(CocoDetection):
    def __init__(self, root, annFile, transform=None, mask_transform=None, random_nb_object=True, max_nb_masks=4, multi_w=False):
        super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        mask = self._load_mask(id)
        if mask is None:
            return None  # Skip this image if no valid mask is available

        img, mask = self.transforms(img, mask)
        return img, mask

    def _load_mask(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  # Return None if there are no annotations

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        # Initialize a list to hold all masks
        masks = []
        if self.random_nb_object and np.random.rand() < 0.5:
            random.shuffle(anns)
            anns = anns[:np.random.randint(1, len(anns)+1)]
        if not(self.multi_w):
            mask = np.zeros((original_height, original_width), dtype=np.float32)
            # one mask for all objects
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m)
            mask = torch.tensor(mask, dtype=torch.float32)
            return mask[None, ...]  # Add channel dimension
        else:
            anns = anns[:self.max_nb_masks]
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            # Stack all masks along a new dimension to create a multi-channel mask tensor
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.bool)
                # Check if the number of masks is less than max_nb_masks
                if masks.shape[0] < self.max_nb_masks:
                    # Calculate the number of additional zero masks needed
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    # Create additional zero masks
                    additional_masks = torch.zeros((additional_masks_count, original_height, original_width), dtype=torch.bool)
                    # Concatenate the original masks with the additional zero masks
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                # Return a tensor of shape (max_nb_masks, height, width) filled with zeros if there are no masks
                masks = torch.zeros((self.max_nb_masks, original_height, original_width), dtype=torch.bool)
            return masks

def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    images, masks = zip(*batch)
    images = torch.stack(images)
    
    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    
    # Pad each mask tensor to have 'max_masks' masks and add the inverse mask
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        union_mask = torch.max(mask, dim=0).values  # Assuming mask is of shape [num_masks, H, W]
        
        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(mask, pad=(0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:
            padded_mask = mask
            
        padded_masks.append(padded_mask)
    
    # Stack the padded masks
    masks = torch.stack(padded_masks)
    
    
    return images, masks

def get_dataloader_segmentation(
    data_dir: str, 
    ann_file: str,
    transform: callable,
    mask_transform: callable,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    random_nb_object = True,
    multi_w=False,
    max_nb_masks = 4
) -> DataLoader:
    """ Get dataloader for COCO dataset. """
    # Initialize the CocoDetection dataset
    dataset = CocoImageIDWrapper(root=data_dir, annFile=ann_file, transform=transform, mask_transform=mask_transform, random_nb_object=random_nb_object, multi_w=multi_w, max_nb_masks=max_nb_masks)

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    
    return dataloader
