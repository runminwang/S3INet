# textnet0.py 表示原来公布、未修改的model

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.vgg import VggNet
from network.resnet import ResNet
from util.config import config as cfg
import time

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class RRGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FNUM = len(cfg.fuc_k)
        self.SepareConv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        channels2 = in_channels + 1
        self.SepareConv1 = nn.Sequential(
            nn.Conv2d(channels2, channels2, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(channels2, channels2, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(channels2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # print("rrgn_in_feature.shape is {}".format(x.shape))

        f_map = list()
        for i in range(self.FNUM):
            if i == 0:
                f = self.SepareConv0(x); f_map.append(f); continue
            b1 = torch.cat([x, f_map[i - 1]], dim=1)
            f = self.SepareConv1(b1)
            # print("after self.SepareConv1.shape is {}".format(f.shape))
            f_map.append(f)
        f_map = torch.cat(f_map, dim=1)
        return f_map


class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', pre_train=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=pre_train)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=pre_train)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            self.merge2 = UpBlok(256 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)
        else:
            print("backbone is not support !")

    def forward(self, x):
        # print("-----------------input_image -----------------")
        # print("input_image.shape is {}".format(x.shape))
        C1, C2, C3, C4, C5 = self.backbone(x)
        # print("-----------------backbone_out_feature -----------------")
        # print("C1.shape is {}".format(C1.shape))
        # print("C2.shape is {}".format(C2.shape))
        # print("C3.shape is {}".format(C3.shape))
        # print("C4.shape is {}".format(C4.shape))
        # print("C5.shape is {}".format(C5.shape))

        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)

        # print("-----------------merge_out_feature -----------------")
        # print("up1.shape is {}".format(up1.shape))
        # print("up2.shape is {}".format(up2.shape))
        # print("up3.shape is {}".format(up3.shape))
        # print("up4.shape is {}".format(up4.shape))
        # print("up5.shape is {}".format(up5.shape))


        return up1, up2, up3, up4, up5


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
        end = time.time()
        up1, up2, up3, up4, up5 = self.fpn(x)
        # print("up1.shape is {}".format(up1.shape))
        # print("up2.shape is {}".format(up2.shape))
        # print("up3.shape is {}".format(up3.shape))
        # print("up4.shape is {}".format(up4.shape))
        # print("up5.shape is {}".format(up5.shape))
        b_time = time.time()-end
        end = time.time()
        predict_out = self.rrgn(up1)
        iter_time = time.time()-end
        
        return predict_out, b_time, iter_time
