#!/bin/bash
for ((i=100;i<=105; i=i+5))
do CUDA_LAUNCH_BLOCKING=1 python3 eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name MLT2017 --checkepoch $i --test_size 256 1920 --batch_size 6 --ratio 1.0 --beta 1.0 --threshold 0.515 --score_i 0.8 --recover watershed --gpu 1 --num_workers 0 --save_dir model/Self_Training/MLT_Total_TASPP_MFPN/mlt_pretraining

#--save_dir model/pretrained/
done
