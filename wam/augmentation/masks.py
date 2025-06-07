# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The following code is lightly adapted from https://github.com/advimman/lama/blob/main/saicinpainting/training/data/masks.py
# The above repository is under Apache-2.0 Licence

import math
import hashlib
from enum import Enum

import os
import cv2
import numpy as np
import os
from PIL import Image

import torch


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part
    

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_len=10, min_width=5, min_times=0, max_times=10, draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = min_len + np.random.randint(max_len)
            brush_w = min_width + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskEmbedder:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_len=60, min_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_len = min_len
        self.min_width = min_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, 
                                          max_len=cur_max_len, max_width=cur_max_width, 
                                          min_len=self.min_len, min_width=self.min_width,
                                          min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)



def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, no_overlap=False):
    height, width = shape
    union_mask = np.zeros((height, width), np.float32)
    
    
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    individual_masks = np.zeros((times, 1, height, width), np.float32)  # Store each rectangle separately

    occupied = np.zeros((height, width), bool)  # To check overlap
    
    for i in range(times):
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            box_width = np.random.randint(bbox_min_size, bbox_max_size + 1)
            box_height = np.random.randint(bbox_min_size, bbox_max_size + 1)
            start_x = np.random.randint(margin, width - margin - box_width + 1)
            start_y = np.random.randint(margin, height - margin - box_height + 1)
            
            if no_overlap:
                # Check if the selected area is free
                if not np.any(occupied[start_y:start_y + box_height, start_x:start_x + box_width]):
                    valid = True
            else:
                valid = True
            
            if valid:
                union_mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
                individual_masks[i, 0, start_y:start_y + box_height, start_x:start_x + box_width] = 1
                if no_overlap:
                    occupied[start_y:start_y + box_height, start_x:start_x + box_width] = True
            attempts += 1
        if not valid:
            print(f"Warning: Could not place non-overlapping rectangle for mask {i + 1}")
    if no_overlap:
        return union_mask[None, ...], individual_masks
    else:
        return union_mask[None, ...]

class RandomRectangleMaskEmbedder:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=1, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None, no_overlap=False, nb_times = None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        if nb_times is not None:
            min_times = nb_times
            max_times = nb_times
        else:
            min_times = self.min_times
            max_times = cur_max_times
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=min_times,
                                          max_times=max_times, no_overlap=no_overlap)



class CustomMaskEmbedder:
    """
    Used to control the size of the rectangles in the mask, between 10 and 100 %.
    """
    def __init__(self):
        #nothing to init
        pass

    def generate_rectangle_masks(self, num_masks=10, num_rectangles=1, image_size=256, min=0.1, max=1):
        """
        Generate rectangle masks for an image.
        Args:
            num_masks (int): Number of masks to generate.
            num_rectangles (int): Number of rectangles in each mask.
            image_size (int): Size of the image.
        Returns:
            list: A list of 2D NumPy arrays representing the masks.
        """
        # Calculate the percentage of area for each rectangle
        percentages = np.linspace(min, max, num_masks)
        # Initialize an empty list to store the masks
        masks = []
        for percentage in percentages:
            mask = np.zeros((image_size, image_size))
            if num_rectangles == 1:
                # Calculate the size of the rectangle
                rect_size = int(np.sqrt(percentage) * image_size)
                # Calculate the start and end indices for the rectangle
                start_idx = (image_size - rect_size) // 2
                end_idx = start_idx + rect_size
                # Create the rectangle in the mask
                mask[start_idx:end_idx, start_idx:end_idx] = 1
            else:
                # Divide the image into subparts
                subpart_width = image_size // num_rectangles
                for i in range(num_rectangles):
                    # Calculate the size of the rectangle
                    rect_width = int(np.sqrt(percentage / num_rectangles) * subpart_width)
                    rect_height = int((percentage / num_rectangles) * image_size)
                    # Calculate the start and end indices for the rectangle
                    start_idx_x = i * subpart_width + (subpart_width - rect_width) // 2
                    end_idx_x = start_idx_x + rect_width
                    start_idx_y = (image_size - rect_height) // 2
                    end_idx_y = start_idx_y + rect_height
                    # Create the rectangle in the mask
                    mask[start_idx_y:end_idx_y, start_idx_x:end_idx_x] = 1
            masks.append(mask)
        return masks

def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


class RandomSuperresMaskEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img, iter_i=None):
        return make_random_superres_mask(img.shape[1:], **self.kwargs)


class DumbAreaMaskEmbedder:
    """
    Generate a mask for an image. 
    The mask is either a random rectangle (if the object is in training mode) or a central square (if the object is not in training mode).
    """
    min_ratio = 0.1
    max_ratio = 0.35
    default_ratio = 0.225

    def __init__(self, is_training):
        self.is_training = is_training

    def _random_vector(self, dimension):
        if self.is_training:
            lower_limit = math.sqrt(self.min_ratio)
            upper_limit = math.sqrt(self.max_ratio)
            mask_side = round((np.random.random() * (upper_limit - lower_limit) + lower_limit) * dimension)
            u = np.random.randint(0, dimension-mask_side-1)
            v = u+mask_side 
        else:
            margin = (math.sqrt(self.default_ratio) / 2) * dimension
            u = round(dimension/2 - margin)
            v = round(dimension/2 + margin)
        return u, v

    def __call__(self, img, iter_i=None, raw_image=None):
        c, height, width = img.shape
        mask = np.zeros((height, width), np.float32)
        x1, x2 = self._random_vector(width)
        y1, y2 = self._random_vector(height)
        mask[x1:x2, y1:y2] = 1
        return mask[None, ...]


class OutpaintingMaskEmbedder:
    def __init__(self, min_padding_percent:float=0.04, max_padding_percent:int=0.25, left_padding_prob:float=0.5, top_padding_prob:float=0.5, 
                 right_padding_prob:float=0.5, bottom_padding_prob:float=0.5, is_fixed_randomness:bool=False):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [left_padding_prob, top_padding_prob, right_padding_prob, bottom_padding_prob]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert len([x for x in [self.min_padding_percent, self.max_padding_percent] if (x>=0 and x<=1)]) == 2, f"Padding percentage should be in [0,1]"
        assert sum(self.probs) > 0, f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4, f"At least one of padding probs is not in [0,1] - {self.probs}"
        if len([x for x in self.probs if x > 0]) == 1:
            print(f"Warning: Only one padding prob is greater than zero - {self.probs}. That means that the outpainting masks will be always on the same side")

    def apply_padding(self, mask, coord):
        mask[int(coord[0][0]*self.img_h):int(coord[1][0]*self.img_h),   
             int(coord[0][1]*self.img_w):int(coord[1][1]*self.img_w)] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent*size)
        n2 = int(self.max_padding_percent*size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash)%(2**32)
        return res

    def __call__(self, img, iter_i=None, raw_image=None):
        c, self.img_h, self.img_w = img.shape
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, f"Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [[
                   (0,0), 
                   (1,self.get_padding(size=self.img_h))
                  ],
                  [
                    (0,0),
                    (self.get_padding(size=self.img_w),1)
                  ],
                  [
                    (0,1-self.get_padding(size=self.img_h)),
                    (1,1)
                  ],    
                  [
                    (1-self.get_padding(size=self.img_w),0),
                    (1,1)
                  ]]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(range(len(coords)), p=np.array(self.probs)/sum(self.probs))
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask[None, ...]


class FullMaskEmbedder:
    def __init__(self, invert_proba=0.5):
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        mask = np.zeros((img.shape[-2], img.shape[-1]), np.float32)
        if self.invert_proba > 0 and np.random.random() < self.invert_proba:
            mask = 1 - mask
        return mask[None, ...]


class CocoSegmentationMaskEmbedder:
    def __init__(self):
        ## Empty class
        pass
        


class MixedMaskEmbedder:
    def __init__(self, irregular_proba=1/4, irregular_kwargs=None,
                 box_proba=1/4, box_kwargs=None,
                 full_proba=1/4, full_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 segm_proba=1/4, segm_kwargs=None,
                 invert_proba=0.5,
                img_size=256,
                 **kwargs
                 ):
        self.probas = []
        self.gens = []

        self.probas.append(irregular_proba)
        if irregular_kwargs is None:
            irregular_kwargs = {'max_angle': 4, 'max_len': 50, 'max_width': 20, 'min_len': 50, 'min_width': 20, 'min_times': 1, 'max_times': 5}
        else:
            irregular_kwargs = dict(irregular_kwargs)
        irregular_kwargs['draw_method'] = DrawMethod.LINE
        self.gens.append(RandomIrregularMaskEmbedder(**irregular_kwargs))

        self.probas.append(box_proba)
        if box_kwargs is None:
            box_kwargs = {'margin': 10, 'bbox_min_size': 30, 'bbox_max_size': 100, 'min_times': 1, 'max_times': 3}
        else:
            box_kwargs = dict(box_kwargs)
        self.gens.append(RandomRectangleMaskEmbedder(**box_kwargs))

        # full mask
        self.probas.append(full_proba)
        if full_kwargs is None:
            full_kwargs = {'invert_proba': 0.0}  # this is handled by the MixedMaskEmbedder later
        self.gens.append(FullMaskEmbedder(**full_kwargs))

        self.probas.append(segm_proba)
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gens.append(CocoSegmentationMaskEmbedder(**segm_kwargs))

        # draw a lot of random squares
        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {'max_angle': 4, 'max_width': 30, 'min_width': 30, 'min_times': 1, 'max_times': 5}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskEmbedder(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskEmbedder(**superres_kwargs))

        if outpainting_proba > 0:
            self.probas.append(outpainting_proba)
            if outpainting_kwargs is None:
                outpainting_kwargs = {}
            self.gens.append(OutpaintingMaskEmbedder(**outpainting_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        

        self.invert_proba = invert_proba

    def __call__(self, 
        imgs, 
        masks = None, 
        iter_i = None, 
        raw_image = None, 
        verbose = False,
        no_overlap = False,
        nb_times = None
    ) -> torch.Tensor:
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        kwargs = {"no_overlap": no_overlap, "nb_times":nb_times} if isinstance(gen, RandomRectangleMaskEmbedder) else {}
        if isinstance(gen, CocoSegmentationMaskEmbedder):
            result = masks
        else:
            if isinstance(gen, RandomRectangleMaskEmbedder) and no_overlap:
                result = gen(imgs[0], iter_i=iter_i, raw_image=raw_image, **kwargs)[1]
            else:
                result = gen(imgs[0], iter_i=iter_i, raw_image=raw_image, **kwargs)
            result = np.repeat(result[np.newaxis, :], imgs.shape[0], axis=0)
            result = torch.from_numpy(result)
        if self.invert_proba > 0 and (np.random.random() < self.invert_proba) and not result.shape[1]>1:
            result = 1 - result
        if verbose:
            print(f"kind = {kind}, result = {result.mean().item()}")
        return result

    def sample_representative_masks(self, img):
        # TODO: Fix issue when probas at 0 and gens are not created
        # Generate masks using the first three generators (assuming they are full, rectangular, and irregular)
        rect_mask = self.gens[1](img)
        irregular_mask = self.gens[0](img)
        full_mask = self.gens[2](img)
        # Generate inverted masks
        inverted_rect_mask = 1 - rect_mask
        inverted_irregular_mask = 1 - irregular_mask
        full_mask = 1-self.gens[2](img)
        # Collect masks into a list
        masks = [full_mask, rect_mask, inverted_rect_mask, irregular_mask, inverted_irregular_mask]
        return torch.tensor(np.stack(masks))

    def sample_multiwm_masks(self, img, nb_times=None):
        # TODO: Fix issue when probas at 0 and gens are not created
        # Generate masks using the first generator (rectangular)
        union, individuals = self.gens[1](img, no_overlap=True, nb_times=nb_times)
        return torch.tensor(union), torch.tensor(individuals)


    def sample_different_sizes(self, size, num_rectangles, num_masks, min, max):
        # Generate masks using the first generator (rectangular)
        generator = CustomMaskEmbedder()
        masks = generator.generate_rectangle_masks(num_masks = num_masks, num_rectangles=num_rectangles, image_size=size, min=min, max=max)
        return torch.tensor(masks)
    
def get_mask_embedder(kind, **kwargs):
    if kind is None:
        kind = "mixed"
    if kwargs is None:
        kwargs = {}

    if kind == "mixed":
        cl = MixedMaskEmbedder
    elif kind == "full":
        cl = FullMaskEmbedder
    elif kind == "outpainting":
        cl = OutpaintingMaskEmbedder
    elif kind == "dumb":
        cl = DumbAreaMaskEmbedder
    else:
        raise NotImplementedError(f"No such embedder kind = {kind}")
    return cl(**kwargs)


if __name__ == "__main__":

    # initialize
    np.random.seed(42)
    mask_embedder = MixedMaskEmbedder(segm_proba=0)
    
    # generate and save 50 masks
    os.makedirs('output', exist_ok=True)
    dummy_img = np.zeros((1, 3, 256, 256))
    for multiple in [1]:
        masks = mask_embedder.sample_different_sizes(256, multiple, 19, 0.05, 0.95)
        for ii in range(masks.shape[0]):
            mask = masks[ii]
            print("area:", masks[ii].mean().item())
            mask = (mask * 255).type(torch.uint8).numpy()
            mask = Image.fromarray(mask)
            mask.save(f'output/mask_multiple={multiple}_number={ii}.png')
    for ii in range(50):
        print(ii)
        # Generate the mask
        mask = mask_embedder(dummy_img, None, verbose=True)
        # Save the mask using PIL
        mask = (mask * 255).type(torch.uint8).numpy()
        mask = Image.fromarray(mask[0, 0])
        mask.save(f'output/mask_{ii}.png')
    union, individuals = mask_embedder.sample_multiwm_masks(dummy_img[0])
    for ii, mask in enumerate(individuals):
        mask = (mask * 255).type(torch.uint8).numpy()
        mask = Image.fromarray(mask[0])
        mask.save(f'output/mask_individual_{ii}.png')
    union = (union * 255).type(torch.uint8).numpy()
    union = Image.fromarray(union[0])
    union.save(f'output/mask_union.png')
        