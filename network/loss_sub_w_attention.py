import torch
import torch.nn as nn
import torch.nn.functional as F
from util.config import config as cfg
from util.basic_loss import DiceLoss
import numpy as np


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # MSE损失计算实际值和预测值之间平方差的平均值（用于回归问题）
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)

        self.KL_loss = torch.nn.KLDivLoss(reduce=False, size_average=False)
        self.k = cfg.fuc_k
        self.dice_loss = DiceLoss(eps=cfg.eps)

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss

    @staticmethod
    def smooth_l1_loss(inputs, target, sigma=9.0, reduction='mean'):
        try:
            diff = torch.abs(inputs - target)
            less_one = (diff < 1.0 / sigma).float()
            loss = less_one * 0.5 * diff ** 2 * sigma \
                   + torch.abs(torch.tensor(1.0) - less_one) * (diff - 0.5 / sigma)
            loss = loss if loss.numel() > 0 else torch.zeros_like(inputs)
        except Exception as e:
            print('smooth L1 Exception:', e)
            loss = torch.zeros_like(inputs)
        if reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction == 'mean':
            loss = torch.mean(loss)
        else:
            loss = loss
        return loss

    def sigmoid_alpha(self, x, d):
        eps = torch.tensor(0.0001)
        alpha = self.k
        dm = torch.where(d >= eps, d, eps)
        betak = (1 + np.exp(-alpha))/(1 - np.exp(-alpha))
        res = (2*torch.sigmoid(x * alpha/dm) - 1)*betak

        return torch.relu(res)

    # def forward(self, inputs, train_mask, tr_mask):
    #     """
    #       calculate textsnake loss
    #     """
    #     b, c, h, w = inputs.shape
    #     # print(f"inputs.shape is {b},{c},{h},{w}")
    #     loss_sum = torch.tensor(0.)
    #     for i in range(c):
    #         reg_loss = self.MSE_loss(torch.sigmoid(inputs[:, i]), tr_mask[:, :, :, i])
    #         # torch.mul对应位相乘
    #         reg_loss = torch.mul(reg_loss, train_mask.float())
    #         reg_loss = self.single_image_loss(reg_loss,  tr_mask[:, :, :, i]) / b
    #         loss_sum = loss_sum + reg_loss
    #
    #     return loss_sum

    def forward(self, inputs, attention, train_mask, tr_mask, expand_mask, train_expand_mask):
        """
          calculate textsnake loss
        """
        b, c, h, w = inputs.shape
        # print(f"inputs.shape is {b},{c},{h},{w}")
        loss_sum = torch.tensor(0.)
        # 概率图损失
        for i in range(c):
            reg_loss = self.MSE_loss(torch.sigmoid(inputs[:, i]), tr_mask[:, :, :, i])
            # torch.mul对应位相乘
            reg_loss = torch.mul(reg_loss, train_mask.float())
            reg_loss = self.single_image_loss(reg_loss,  tr_mask[:, :, :, i]) / b
            loss_sum = loss_sum + reg_loss

        # 分割图损失
        seg_loss = self.dice_loss(attention, expand_mask, train_expand_mask)

        # # 原有损失函数权重
        # config.alpha = 1
        # # 辅助损失函数权重
        # config.beta = 0.5

        aux_loss = cfg.beta * seg_loss

        # cfg文件中添加相关参数
        loss_all = cfg.alpha * loss_sum + aux_loss


        return loss_all, aux_loss

