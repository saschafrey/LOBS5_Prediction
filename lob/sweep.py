#import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"
#import torch
#torch.multiprocessing.set_start_method('spawn')
import wandb
from lob.train import train


sweep_config = {
    'program': '/nfs/home/peern/LOBS5/run_train.py',
    'method': 'random',
    'metric': {
        'name': 'Val loss',
        'goal': 'minimize'
    },
    'early_terminate': {  # https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#early_terminate
        'type': 'hyperband',  # https://arxiv.org/abs/1603.06560
        'min_iter': 3,
        'eta': 3,
    },
    'parameters': {
        'USE_WANDB': {'value': True},
        'wandb_project': {'value': 'LOBS5'},
        'wandb_entity': {'value': 'peer-nagy'},
        'dir_name': {'value': './data'},
        'dataset': {'value': 'lobster-prediction'},
        'masking': {'value': 'causal'},
        'use_book_data': {'value': True},
        'restore': {'value': ''},
        'restore_step': {'value': 0},

        'n_message_layers': {'values': [2, 3, 4]},
        'n_book_pre_layers': {'values': [1, 2]},
        'n_book_post_layers': {'values': [1, 2]},
        'n_layers': {'values': [2,3,4,5,6]},

        'd_model': {'values': [16, 32]},
        'ssm_size_base': {'values': [16, 32, 64]},
        'blocks': {'values': [4, 8]},
        'C_init': {'values': ["trunc_standard_normal", "lecun_normal", "complex_normal"]},
        'discretization': {'values': ["zoh", "bilinear"]},
        'mode': {'values': ["pool", "last"]},
        'activation_fn': {'values': ["full_glu", "half_glu1", "half_glu2", "gelu"]},
        'conj_sym': {'values': [True, False]},
        'clip_eigs': {'values': [True, False]},
        'bidirectional': {'values': [True, False]},
        'dt_min': {'value': 0.001},
        'dt_max': {'value': 0.1},
        
        'prenorm': {'values': [True, False]},
        'batchnorm': {'values': [True, False]},
        'bn_momentum': {'min': 0.1, 'max': 0.99},
        'bsz': {'values': [8, 16]},
        'epochs': {'value': 30},
        'early_stop_patience': {'value': 1000},  # handle early stopping in sweep
        'ssm_lr_base': {'min': 1e-6, 'max': 2e-3, 'distribution': 'log_uniform_values'},
        'lr_factor': {'min': 0.1, 'max': 2.0, 'distribution': 'uniform'},
        'lr_min': {'value': 0},
        'cosine_anneal': {'value': True},
        'warmup_end': {'value': 1},
        'lr_patience': {'value': 1000000},
        'reduce_factor': {'value': 1},
        'p_dropout': {'min': 0., 'max': 0.5},
        'weight_decay': {'min': 0., 'max': 0.5},
        'opt_config': {'values': ['standard', 'BandCdecay', 'BfastandCdecay', 'noBCdecay']},
        'jax_seed': {'value': 42},
    }
}

if __name__ == "__main__":

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="LOBS5",
        entity="peer-nagy"
    )
    #print('sweep_id', sweep_id)
