#!/bin/bash

/home/sascha/miniconda3/envs/s5_lob/bin/python /data1/sascha/LOBS5Prediction/run_train.py --USE_WANDB false --dataset 'FI-2010-classification' --msg_seq_len 100 --horizon_type messages --prediction_horizon 10 --dir_name /data1/sascha/data --num_devices 1