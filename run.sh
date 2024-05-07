#!/bin/bash

/home/sascha/miniconda3/envs/s5_lob/bin/python /data1/sascha/LOBS5Prediction/run_train.py --USE_WANDB true --dataset 'FI-2010-classification' --msg_seq_len 100 --horizon_type messages --prediction_horizon 50 --dir_name /data1/sascha/data --num_devices 4 --use_book_only True --use_book_data False --ssm_lr_base 1e-4 --bsz 64 --p_dropout 0.5 --n_layers 4 --warmup_end 5 --n_book_pre_layers 0 --zero_sequences True
