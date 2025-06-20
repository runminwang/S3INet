# textnet0.py 表示原来公布、未修改的model

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VggNet
from network.resnet import ResNet
from util.config import config as cfg
import time
from collections import OrderedDict
import torchvision
import math



# 创新定稿的MergeFpn（已完成）
from network.textnet_merge_fpn_DFFC import FPN
# 针对MergeFpn中多尺度语义信息DASPP模块的消融（已完成）
# from network.textnet_merge_fpn_DFFC_wp54 import FPN

# 针对MergeFpn中不使用parallel-Merge的消融（已完成）
# from network.textnet_merge_fpn_DFFC_w_cat_merge import FPN

# 针对MergeFpn中不使用前景注意力的消融（已完成）
# from network.textnet_merge_fpn_DFFC_wo_attention import FPN



# 创新定稿的TASPP（已完成）
from network.textnet_cs_rate3_TASPP_add_pcon1 import RRGN

# 针对TASPP中级联方式的消融 （已完成）
# from network.textnet_TASPP_add_cscon1 import RRGN
# 针对TASPP中不使用多尺度的消融 （已完成）
# from network.textnet_TASPP_add_pcon1_woms import RRGN
# 针对TASPP中仅使用垂直方向卷积的消融 （已完成）
# from network.textnet_TASPP_add_pcon1_woh import RRGN


class TextNet(nn.Module):
    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, pre_train=is_training)
        self.rrgn = RRGN(16)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x):
    # 有前景注意力的FPN!!!!!
        end = time.time()
        # up1, up2, up3, up4, up5 = self.fpn(x)
        feature_total, attention = self.fpn(x)
        # print("feature_total.shape is {}".format(feature_total.shape))
        # print("attention.shape is {}".format(attention.shape))
        # print("up3.shape is {}".format(up3.shape))
        # print("up4.shape is {}".format(up4.shape))
        # print("up5.shape is {}".format(up5.shape))

        mlt = torch.mul(feature_total, attention)
        out = torch.add(feature_total, mlt)
        b_time = time.time() - end

        end = time.time()
        predict_out = self.rrgn(out)
        iter_time = time.time() - end

        return predict_out, attention, b_time, iter_time
