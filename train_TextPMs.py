import os
import gc
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from dataset import SynthText, TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.loss import TextLoss

from network.textnet0 import TextNet

from util.augmentation import Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR
# 可视化loss图像
from util.plot import plot_loss
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'TextPMs_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        #'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


# def load_model(model, model_path):
#     print('Loading from {}'.format(model_path))
#     state_dict = torch.load(model_path)
#     print("model.pth.state_dict is :")
#     params = state_dict  # 获得模型的原始状态以及参数。
#     for k in params['model.fpn']:
#         print(k)  # 只打印key值，不打印具体参数。
#         # lr
#         # epoch
#         # model
#         # optimizer
#     model.load_state_dict(state_dict['model'])

def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


# 针对一个epoch的训练
def train(model, train_loader, criterion, scheduler, optimizer, epoch, logger=None):

    global train_step
    losses = AverageMeter()
    model.train()
    # scheduler.step()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))
    for i, (img, train_mask, tr_mask) in enumerate(train_loader):
        train_step += 1
        img, train_mask, tr_mask = to_device(img, train_mask, tr_mask)
        output, _, _ = model(img)
        loss = criterion(output, train_mask, tr_mask)
        # backward
        try:
            optimizer.zero_grad()
            loss.backward()
        except:
            print("loss gg")
            continue

        optimizer.step()
        losses.update(loss.item())
        gc.collect()

        if cfg.viz and i % cfg.viz_freq == 0:
            visualize_network_output(output, tr_mask, mode='train')

        if i % cfg.display_freq == 0:
            print('({:d} / {:d})  Loss: {:.4f} '
                  .format(i, len(train_loader), loss.item()))

        if i % cfg.log_freq == 0:
            logger.write_scalars({
                'loss': loss.item(),
            }, tag='train', n_iter=train_step)

    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))

    # 添加对losses保存、后续读取画图的代码
    # Losses_avg = np.array(losses.avg)
    # # np.save('loss_data/MLT_Pretrain_resume64/loss2/epoch_{}'.format(epoch), Losses_avg)
    # np.save('loss_data/Original_Model/Totaltext_no_pretraining/loss/epoch_{}'.format(epoch), Losses_avg)

def main():

    global lr
    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        # valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='data/SynthText',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Icdar2015':
        trainset = Icdar15Text(
            data_root='data/Icdar2015/Icdar2015',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.exp_name == 'MLT2017':
        trainset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500':
        trainset = TD500Text(
            data_root='data/TD500',
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    else:
        print("dataset name is not correct")

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
    logger = LogSummary(log_dir)

    # Model
    model = TextNet(backbone=cfg.net, is_training=True)
    if cfg.mgpu:
        model = nn.DataParallel(model)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    if cfg.resume:
        load_model(model, cfg.resume)

#     if cfg.resume:
#         load_model(model,lr,cfg.resume)

    criterion = TextLoss()

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    print('Start training TextPMs.')
    # for epoch in range(cfg.start_epoch, cfg.start_epoch + cfg.max_epoch+1):
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        scheduler.step()
        # 针对一个epoch的训练
        train(model, train_loader, criterion, scheduler, optimizer, epoch, logger=logger)

    # 所有epoch训练完后，对可视化loss曲线
    # plot_loss(cfg.max_epoch)

    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()

