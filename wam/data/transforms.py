# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import transforms

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

normalize_img = transforms.Normalize(image_mean, image_std)
unnormalize_img = transforms.Normalize(-image_mean / image_std, 1 / image_std)
unstd_img = transforms.Normalize(0, 1 / image_std)
std_img = transforms.Normalize(0, image_std)
imnet_to_lpips = transforms.Compose([
    unnormalize_img,
])

default_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_img,
])

def rgb_to_yuv(img):
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.14713, -0.28886, 0.436],
                      [0.615, -0.51499, -0.10001]], dtype=torch.float32).to(img.device)
    img = img.permute(0, 2, 3, 1)  # b h w c
    yuv = torch.matmul(img, M)
    yuv = yuv.permute(0, 3, 1, 2)
    return yuv

def yuv_to_rgb(img):
    M = torch.tensor([[1.0, 0.0, 1.13983],
                      [1.0, -0.39465, -0.58060],
                      [1.0, 2.03211, 0.0]], dtype=torch.float32).to(img.device)
    img = img.permute(0, 2, 3, 1)  # b h w c
    rgb = torch.matmul(img, M)
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb

def get_transforms(
    img_size: int,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform
    
def get_transforms_segmentation(
    img_size: int,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
):
    """
        Get transforms for segmentation task.
        Important: No random geometry transformations must be applied for the mask to be valid.
        Args:
            img_size: int: size of the image
            brightness: float: brightness factor
            contrast: float: contrast factor
            saturation: float: saturation factor
            hue: float: hue factor
        Returns:
            train_transform: transforms.Compose: transforms for training set
            train_mask_transform: transforms.Compose: transforms for mask in training set
            val_transform: transforms.Compose: transforms for validation set
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])  
    train_mask_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size)
    ])  
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize_img,
    ])
    val_mask_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size)
    ])
    return train_transform, train_mask_transform, val_transform, val_mask_transform
