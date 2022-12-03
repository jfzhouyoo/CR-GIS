#!/bin/bash

set -e

pythonpath='python'

# Dataset: TgReDial

##################################################
####          Only recommend movies.          ####
##################################################

# PRETRAIM Goal-oriented Fusion Mechanism
${pythonpath} main.py --dataset tgredial --train_rec_only True --use_mim True --is_pretrain_mim True --mim_epochs 10 --encoder_num_layers 2 --log_freq_iter 50 --gpu 1

# TRAIN & EVAL Goal-oriented Recommendation
${pythonpath} main.py --dataset tgredial --train_rec_only True --use_mim True --is_load_mim True --is_training_rec True --eval_rec True --encoder_num_layers 2 --weight_mim 0.1 --rec_epochs 100 --log_freq_iter 50 --gpu 1


##################################################
####   Topics are also recommendation goals.  ####
##################################################

# PRETRAIM Goal-oriented Fusion Mechanism
${pythonpath} main.py --dataset tgredial --use_mim True --is_pretrain_mim True --mim_epochs 10 --encoder_num_layers 2 --gpu 1

# TRAIN & EVAL Goal-oriented Recommendation
${pythonpath} main.py --dataset tgredial --use_mim True --is_load_mim True --is_training_rec True --eval_rec True --encoder_num_layers 2 --weight_mim 0.1 --rec_epochs 100 --gpu 1

# TRAIN & EVAL  Goal-aware Response Generation
${pythonpath} main.py --dataset tgredial --use_mim True --is_training_con True --is_generator True --freeze_rec True --con_epochs 90 --weight_rec 0.0 --weight_mim 0.00 --joint_rec False --joint_mim False --encoder_num_layers 2 --history_window_size 5 --gpu 1 # --is_load_con True

# ${pythonpath} eval_generation.py
