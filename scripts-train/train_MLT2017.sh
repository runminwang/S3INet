#!/bin/bash
cd ../

CUDA_LAUNCH_BLOCKING=1 python train_TextPM_cs_rate3_mfpn_dffc_w_attention.py --exp_name MLT2017 --net resnet50 --optim SGD --lr 0.01 --input_size 640 --batch_size 6 --gpu 0 --max_epoch 300 --num_workers 24 --ratio 1.0 --beta 1.0 --start_epoch 0 --save_freq 2 --resume model/publish_pretrained/MLT2017/TextPMs_resnet50_64.pth --save_dir model/Self_Training/MLT_Total_TASPP_MFPN/mlt_pretraining # --viz