python run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=8 --bsz=12 --d_model=128 --dataset=lobster-prediction \
                    --dir_name='./data/new_enc' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=200 --jax_seed=42 --lr_factor=1 --n_layers=6 \
                    --opt_config=standard --p_dropout=0.2 --ssm_lr_base=0.0005 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=random \
                    --num_devices=1 --n_data_workers=0
