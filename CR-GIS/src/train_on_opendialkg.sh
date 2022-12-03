#!/bin/bash

set -e

pythonpath='python'

# Dataset: OpenDialKG

# PRETRAIM Goal-oriented Fusion Mechanism
${pythonpath} main.py --dataset opendialkg --use_mim True --is_pretrain_mim True --mim_epochs 10 --encoder_num_layers 2 --log_freq_iter 50 --gpu 1

# TRAIN & EVAL Goal-oriented Recommendation
${pythonpath} main.py --dataset opendialkg --use_mim True --is_load_mim True --is_training_rec True --eval_rec True --weight_mim 0.1 --rec_epochs 100 --encoder_num_layers 2 --log_freq_iter 50  --gpu 1

# TRAIN & EVAL  Goal-aware Response Generation
${pythonpath} main.py --dataset opendialkg --use_mim True --is_training_con True --is_generator True --con_epochs 90 --weight_rec 0.05 --weight_mim 0.005 --joint_rec True --joint_mim True  --encoder_num_layers 2 --history_window_size 5 --log_freq_iter 50 --gpu 1

