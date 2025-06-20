# S3INet
This is a Pytorch implementation of "S3INet: Semantic-Information Space Sharing Interaction Network for Arbitrary Shape Text Detection".
![overall2](https://github.com/user-attachments/assets/1ad02a7d-be68-43b0-90f0-a3d800e08f07)

# Prerequisites
python 3.7；
PyTorch 1.7.0；
Numpy >=1.16；
CUDA >=10.2；
GCC >=9.0；
opencv-python < 4.5.0；
NVIDIA GPU  3090；

# Dataset Links
NOTE: The images of each dataset can be obtained from their official website.

# Training
Runing the training scripts
We provide training scripts for each dataset in scripts-train, such as Total-Text, CTW-1500,MSRA-TD500. We also provide pre-training script of ICDAR2017-MLT ...

# Running Evaluation
run:
sh eval.sh

You can evaluate models individually or in a loop. The details in a are as follows:
----
#!/bin/bash
###### test eval ############
##################### Total-Text ###################################
CUDA_LAUNCH_BLOCKING=1 python eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Totaltext --checkepoch 250 --test_size 640 1024 --threshold 0.4 --score_i 0.7 --recover watershed --gpu 0 # --viz


###################### CTW-1500 ####################################
CUDA_LAUNCH_BLOCKING=1 python eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Ctw1500 --checkepoch 480 --test_size 512 1024 --threshold 0.4 --score_i 0.7 --recover watershed --gpu 0

for ((i=0; i<=300; i=i+5))
do CUDA_LAUNCH_BLOCKING=1 python3 eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Ctw1500 --checkepoch $i --test_size 512 1024 --batch_size 4 --ratio 1.0 --beta 1.0 --threshold 0.4 --score_i 0.7 --recover watershed --gpu 0 --num_workers 0 --save_dir model/Self_Training/B+SSM+MPACM/finetune_base_mlt112/
done


#################### MSRA-TD500 ######################################
CUDA_LAUNCH_BLOCKING=1 python eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name TD500 --checkepoch 125 --test_size 0 832 --threshold 0.45 --score_i 0.835 --recover watershed --gpu 0


#################### Icdar2015 ######################################
CUDA_LAUNCH_BLOCKING=1 python eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Icdar2015 --checkepoch 370 --test_size 960 1920 --threshold 0.515 --score_i 0.814 --recover watershed --gpu 0

for ((i=155; i<=155; i=i+5))
do CUDA_LAUNCH_BLOCKING=1 python3 eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Icdar2015 --checkepoch $i --test_size 960 1920 --batch_size 4 --ratio 1.0 --beta 1.0 --threshold 0.515 --score_i 0.814 --recover watershed --gpu 0 --num_workers 0
done

####################  CTST ######################################
for ((i=200; i<=300; i=i+5))
do CUDA_LAUNCH_BLOCKING=1 python3 eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Icdar2015 --checkepoch $i --test_size 960 1920 --batch_size 4 --ratio 1.0 --beta 1.0 --threshold 0.515 --score_i 0.86 --recover watershed --gpu 0 --num_workers 0
done

####################  TPD ######################################
for ((i=200; i<=315; i=i+5))
do CUDA_LAUNCH_BLOCKING=1 python3 eval_TextPMs_CS_Rate3_merge_fpn_DFFC.py --exp_name Icdar2015 --checkepoch $i --test_size 960 1920 --batch_size 4 --ratio 1.0 --beta 1.0 --threshold 0.515 --score_i 0.85 --recover watershed --gpu 0 --num_workers 0
done

NOTE：If you want to save the visualization results, you need to open “--viz”.


# Visualization
For more detailed visualization results, please refer to the paper.
