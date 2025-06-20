
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
        f_map = list()
        for i in range(self.FNUM):
            if i == 0:
                f = self.SepareConv0(x);
                f_map.append(f);
                continue
            b1 = torch.cat([x, f_map[i - 1]], dim=1)
            f = self.SepareConv1(b1)
            f_map.append(f)
        f_map = torch.cat(f_map, dim=1)
        return f_map



# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=False,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))
    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # output_stride = cfg.MODEL.OUTPUT_STRIDE
        output_stride = 16
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


# 针对cat之后进行3*3深度可分离处理
class _CatSpeHead(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, norm_layer=nn.BatchNorm2d, relu_first=False):
        super(_CatSpeHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            SeparableConv2d(in_channels, inter_channels, kernel_size, norm_layer=norm_layer, relu_first=relu_first),
            SeparableConv2d(inter_channels, channels, kernel_size, norm_layer=norm_layer, relu_first=relu_first))

    def forward(self, x):
        return self.block(x)





# 产生文本分割图
class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            # SeparableConv2d(in_channels, inter_channels,kernel_size=3, dilation=1, relu_first=False),
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


# 上下文交叉语义信息自适应融合
# 结合针对分类保留语义信息的处理方法对高层特征进行梳理，并与底层特征融合，得到具有丰富多层次上下文的输出特征。
class Context_Semantics(nn.Module):
    def __init__(self):
        super(Context_Semantics, self).__init__()
        # 特征下采样2倍，得到显著目标的语义信息,Gcls = Concat(下采样pl,pl+1),最终尺寸为pl的1/2

        # 对 P45进行这样处理，得到最终P5的强语义融合特征，之后最终于P123Merge融合，得到大小为P1的融合特征图
        self.downscale_4 = self._make_layer33(128, 128, 3, 2, 1)

        self.P5_1 = self._make_layer33(384, 16, 3, 1, 1)

        self.P5_2 = self._make_layer33(384, 32, 3, 1, 1)

        self.P5_3 = self._make_layer33(384, 64, 3, 1, 1)

        self.aspp = _ASPP(384, 384)

        self.P11 = self._make_layer33(16, 16, 3, 1, 1)
        self.P12 = self._make_layer(32, 16, 1, 1, 0)
        self.P13 = self._make_layer(64, 16, 1, 1, 0)

        self.P21 = self._make_layer33(16, 32, 3, 2, 1)
        self.P22 = self._make_layer33(32, 32, 3, 1, 1)
        self.P23 = self._make_layer(64, 32, 1, 1, 0)

        self.P31 = self._make_layer_233(16, 64, 3, 2, 1)  # 下采样4倍实现
        self.P32 = self._make_layer33(32, 64, 3, 2, 1)
        self.P33 = self._make_layer33(64, 64, 3, 1, 1)

        self.ConvLinear2 = _CatSpeHead(112, 16, 3)


        self.fcn = _FCNHead(16, 2)

        self.m = nn.Softmax(dim=1)  # [3, 1, 200, 304]  # 结果只有一个种类（文本）前景的概率
        # self.m = nn.Sigmoid()   # [3, 1, 200, 304]    # 将多个种类的结果限制在0-1


    def _upsample(self, x,scale):
        _,_,h,w=x.size()
        return F.interpolate(x, size=(h*scale, w*scale), mode='bilinear', align_corners=True)

    # 正常卷积Conv2d
    def _make_layer(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
                nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
                nn.BatchNorm2d(num_channels_cur_layer),
                nn.ReLU(inplace=True),
                ))
        return nn.Sequential(*transition_layers)

    # 3*3深度可分离卷积
    def _make_layer33(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
                SeparableConv2d(num_channels_pre_layer, num_channels_cur_layer, kerner_size,stride,padding,relu_first=False),
                nn.BatchNorm2d(num_channels_cur_layer),
                nn.ReLU(inplace=True),
                ))
        return nn.Sequential(*transition_layers)

    # 3*3深度可分离卷积,下采样4倍
    def _make_layer_233(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
            # nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
            SeparableConv2d(num_channels_pre_layer, num_channels_cur_layer, kerner_size, stride, padding,
                            relu_first=False),
            nn.BatchNorm2d(num_channels_cur_layer),
            nn.ReLU(inplace=True),
            SeparableConv2d(num_channels_cur_layer, num_channels_cur_layer, kerner_size, stride, padding,
                            relu_first=False),
            nn.BatchNorm2d(num_channels_cur_layer),
        ))
        return nn.Sequential(*transition_layers)

    # 3*3深度可分离卷积,下采样8倍
    def _make_layer_333(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
            SeparableConv2d(num_channels_pre_layer, num_channels_pre_layer, kerner_size, stride, padding,
                            relu_first=False),
            nn.BatchNorm2d(num_channels_pre_layer),
            nn.ReLU(inplace=True),
            SeparableConv2d(num_channels_pre_layer, num_channels_cur_layer, kerner_size, stride, padding,
                            relu_first=False),
            nn.BatchNorm2d(num_channels_cur_layer),
            nn.ReLU(inplace=True),
            SeparableConv2d(num_channels_cur_layer, num_channels_cur_layer, kerner_size, stride, padding,
                            relu_first=False),
            nn.BatchNorm2d(num_channels_cur_layer)
        ))
        return nn.Sequential(*transition_layers)


    def forward(self, x):
        p1, p2, p3, p4, p5 = x
        # 16、32、64、128、256 对应p1, p2, p3, p4, p5通道数
        # 对P345处理得到高层语义信息

        p5_4 = torch.cat([self.downscale_4(p4), p5], 1)  # p5大小* 384
        # 将p5_4大小* 384 变化为适应p1, p2, p3，与其相加融合。

        p5_4 = self.aspp(p5_4)
        #
        #
        p1_5 = self._upsample(self.P5_1(p5_4), scale=16)
        p2_5 = self._upsample(self.P5_2(p5_4), scale=8)
        p3_5 = self._upsample(self.P5_3(p5_4), scale=4)

        p1 = p1_5 + p1
        p2 = p2_5 + p2
        p3 = p3_5 + p3

        # 使用RFN中merge进行融合，之后进行分割
        # 统一为p1的尺寸
        p11 = self.P11(p1)
        p12 = self._upsample(self.P12(p2), scale=2)
        p13 = self._upsample(self.P13(p3), scale=4)

        # 统一为p2的尺寸
        p21 = self.P21(p1)
        p22 = self.P22(p2)
        p23 = self._upsample(self.P23(p3), scale=2)

        # 统一为p3的尺寸
        p31 = self.P31(p1)
        p32 = self.P32(p2)
        p33 = self.P33(p3)

        s1 = p11 + p12 + p13
        s2 = p21 + p22 + p23
        s3 = p31 + p32 + p33

        # print("s1.size() is{}".format(s1.shape))
        # print("s2.size() is{}".format(s2.shape))
        # print("s3.size() is{}".format(s3.shape))

        x0_h, x0_w = p1.size(2), p1.size(3)

        # print("p2.size(2) p2.size(3) is{}".format(x0_h,x0_w))
        # s2 = F.upsample(s2, size=(x0_h, x0_w), mode='bilinear')
        s2 = F.upsample(s2, size=(x0_h, x0_w), mode='bilinear')
        s3 = F.upsample(s3, size=(x0_h, x0_w), mode='bilinear')

        # cat融合,使用3*3深度可分离卷积降维块处理
        low_feature = self.ConvLinear2(torch.cat([s1, s2, s3], 1))

        # 对其不使用前景文本目标感知进行增强，直接输出融合的底层特征
        return low_feature

        # 产生前景文本目标感知
        # out = self.fcn(low_feature)
        # text_prob = self.m(out)
        # # 因为要在训练过程中对其进行损失计算，所以返回相应注意力图，exp（）加大前景与背景的差距
        # attention = text_prob[:, 1, :, :].unsqueeze(1).exp()  # attention.shape is torch.Size([3, 1, 200, 304])
        # # print("attention.shape is {}".format(attention.shape))
        # return low_feature, attention


# 修改之后
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

            self.cs = Context_Semantics()

        else:
            print("backbone is not support !")

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        # 原有的FPN处理
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)

        input_features = []

        input_features.append(up1)
        input_features.append(up2)
        input_features.append(up3)
        input_features.append(up4)
        input_features.append(up5)

        # feature_total, attention = self.cs(input_features)
        feature_total = self.cs(input_features)
        return feature_total

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
        # up1, up2, up3, up4, up5 = self.fpn(x)


        # feature_total, attention = self.fpn(x)

        feature_total = self.fpn(x)


        # print("feature_total.shape is {}".format(feature_total.shape))
        # print("attention.shape is {}".format(attention.shape))
        # print("up3.shape is {}".format(up3.shape))
        # print("up4.shape is {}".format(up4.shape))
        # print("up5.shape is {}".format(up5.shape))

        # mlt = torch.mul(feature_total, attention)
        # out = torch.add(feature_total, mlt)
        b_time = time.time() - end

        end = time.time()
        predict_out = self.rrgn(feature_total)
        iter_time = time.time() - end

        # return predict_out, attention, b_time, iter_time

        # 已没有注意力权重，直接去除
        return predict_out, b_time, iter_time
