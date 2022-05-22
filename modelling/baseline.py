# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import imp
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from utils.misc import get_backbone

from .backbones.resnet import BasicBlock, Bottleneck, ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.patchconvnet import S60
from .backbones.cait import cait_XS24, cait_XXS36
from .backbones.vit import deit_small_patch16_LS

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, cfg):
        super(Baseline, self).__init__()

        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAINED
        self.use_mixed_precision = cfg.USE_MIXED_PRECISION

        self.gap = nn.AdaptiveAvgPool2d(1)
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(last_stride)
        elif "patchconvnet_s60" in model_name:
            self.base = S60(emb_size=cfg.MODEL.BACKBONE_EMB_SIZE)
            self.gap = nn.Identity()
        elif model_name == "cait_XS24":
            self.base = cait_XS24(pretrained=cfg.MODEL.PRETRAINED,
                                  last_emb_size=cfg.MODEL.BACKBONE_EMB_SIZE)
            self.gap = nn.Identity()
        elif model_name == "cait_XXS36_224":
            self.base = cait_XXS36(pretrained=cfg.MODEL.PRETRAINED, model_name="XXS36_224",
                                   last_emb_size=cfg.MODEL.BACKBONE_EMB_SIZE)
            self.gap = nn.Identity()
        elif model_name == "deit_small_224_21k":
            self.base = deit_small_patch16_LS(pretrained=cfg.MODEL.PRETRAINED, img_size=cfg.INPUT.SIZE_TRAIN[0],
                                              last_emb_size=cfg.MODEL.BACKBONE_EMB_SIZE)
            self.gap = nn.Identity()
            
        
            
        self.model_name = model_name

        if pretrain_choice and not cfg.MODEL.RESUME_TRAINING and not cfg.TEST.ONLY_TEST:
            # If resume training do not load backbone weights
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')


    def forward(self, x):
        base_out = self.base(x)
        global_feat = self.gap(base_out)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        
        return base_out, global_feat

    def load_param(self, trained_path, load_specific=None):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if load_specific is not None:
                if load_specific in i:
                    self.state_dict()[i].copy_(param_dict[i])
            else:
                if 'classifier' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])
