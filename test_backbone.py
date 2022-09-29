import numpy as np
import os.path as osp
from PIL import Image
from train_ctl_model import CTLModel
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
from torchsummary import summary
from modelling.backbones.patchconvnet import S60
from modelling.backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from modelling.backbones.vit import deit_small_patch16_LS, deit_small_patch16_36_LS



    
model = CTLModel.load_from_checkpoint("./logs/df1/320_deit_small/train_ctl_model/version_102/checkpoints/epoch=26.ckpt",
                                      learning_rate=0.1)
print(opt)