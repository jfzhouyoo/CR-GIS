#!/bin/bash

set -e

pythonpath='python'

# pretrain mim
# ${pythonpath} main.py --use_mim True --is_pretrain_mim True --mim_epochs 10 --encoder_num_layers 2

# train and evaluate recommendation

# -------------------
# ${pythonpath} main.py --use_mim True --is_load_mim True --is_training_rec True --eval_rec True --encoder_num_layers 2 --weight_mim 0.1 --rec_epochs 100
# -------------------

# train and evaluate conversation

# -------------------
${pythonpath} main.py --use_mim True --is_training_con True --is_generator True --con_epochs 90 --weight_rec 0.1 --weight_mim 0.01 --joint_rec True --joint_mim True --encoder_num_layers 2 #--history_window_size 5
# ${pythonpath} main.py --use_mim True --is_training_con True --is_generator True --con_epochs 10 --weight_rec 0.05 --weight_mim 0.005 --joint_rec True --joint_mim True --encoder_num_layers 2 --is_load_con True
# -------------------

# eval recommendation
${pythonpath} main.py --use_mim True --is_load_mim True --eval_rec True --weight_mim 0.1 --rec_epochs 30 --is_load_con True --encoder_num_layers 2

