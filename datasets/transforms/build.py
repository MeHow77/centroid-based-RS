# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .random_erasing import RandomErasing

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms

from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor

import numpy as np
from torchvision import datasets, transforms
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
 
    
    
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
class ColorJitter(object):
    """
    Apply ColorJitter to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = T.ColorJitter(0.3, 0.3, 0.3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img      
    

class ReidTransforms():

    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self, is_train=True):
        normalize_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        if is_train:
            transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                #T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                T.RandomChoice([gray_scale(p=0.4),
                                Solarization(p=0.3),
                                GaussianBlur(p=0.7)]),
                T.ToTensor(),
                normalize_transform,
                #RandomErasing(probability=self.cfg.INPUT.RE_PROB, mean=self.cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])

        return transform