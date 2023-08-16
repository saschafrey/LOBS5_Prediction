import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

import torch
torch.multiprocessing.set_start_method('spawn')

# Add parent folder to path (to run this file from subdirectories)
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(parent_folder_path)

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))

from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderbook as job

from argparse import Namespace
from glob import glob
import numpy as onp
import pandas as pd
from functools import partial
from typing import Union, Optional
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.nn import one_hot
from jax import random
from jax.scipy.linalg import block_diag
from flax import jax_utils
from flax.training import checkpoints
import orbax

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.dataloading import make_data_loader
from lob_seq_model import LobPredModel
from encoding import Vocab, Message_Tokenizer
from lobster_dataloader import LOBSTER_Dataset, LOBSTER_Subset, LOBSTER_Sampler, LOBSTER

import preproc
import inference
import validation_helpers as valh
from lob.init_train import init_train_state, load_checkpoint, load_args_from_checkpoint
import lob.encoding as encoding

##################################################

# get args from command line to select stock between GOOG, NFLX, INTC
parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, default='GOOG', help='stock to evaluate')
args = parser.parse_args()

if args.stock == 'GOOG':
    ckpt_path = '../checkpoints/treasured-leaf-149_84yhvzjt/' # 0.5 y GOOG, (full model)
    data_dir = '/nfs/home/peern/LOBS5/data/test_set/GOOG/'
    save_dir = '/nfs/home/peern/LOBS5/data/results/GOOG/inference/'
elif args.stock == 'NFLX':
    ckpt_path = '../checkpoints/fanciful-grass-151_q36k51ii/' # 0.5 y NFLX, (full model)
    data_dir = '/nfs/home/peern/LOBS5/data/test_set/NFLX/'
    save_dir = '/nfs/home/peern/LOBS5/data/results/NFLX/inference/'
elif args.stock == 'INTC':
    ckpt_path = '../checkpoints/pleasant-cherry-152_i6h5n74c/' # 0.5 y INTC, (full model)
    data_dir = '/nfs/home/peern/LOBS5/data/test_set/INTC/'
    save_dir = '/nfs/home/peern/LOBS5/data/results/INTC/inference/'
else:
    raise ValueError('Invalid stock name')

##################################################

n_messages = 500
v = Vocab()
n_classes = len(v)
seq_len = n_messages * Message_Tokenizer.MSG_LEN
book_dim = 501 #b_enc.shape[1]
book_seq_len = n_messages

rng = jax.random.PRNGKey(42)
rng, rng_ = jax.random.split(rng)

args = load_args_from_checkpoint(ckpt_path)

# scale down to single GPU, single sample inference
args.bsz = 1
args.num_devices = 1

batchnorm = args.batchnorm

# load train state from disk

new_train_state, model_cls = init_train_state(
    args,
    n_classes=n_classes,
    seq_len=seq_len,
    book_dim=book_dim,
    book_seq_len=book_seq_len,
)

ckpt = load_checkpoint(
    new_train_state,
    ckpt_path,
    args.__dict__)
par_state = ckpt['model']

# deduplicate params (they get saved per gpu in training)
state = par_state.replace(
    params=jax.tree_map(lambda x: x[0], par_state.params),
    batch_stats=jax.tree_map(lambda x: x[0], par_state.batch_stats),
)

model = model_cls(training=False, step_rescale=1.0)

##################################################

import lob.evaluation as eval
from lob.preproc import transform_L2_state

n_gen_msgs = 100  #500 # how many messages to generate into the future
n_messages = 500
n_eval_messages = 100  # how many to load from dataset 
eval_seq_len = n_eval_messages * Message_Tokenizer.MSG_LEN

data_levels = 10
sim_book_levels = 20 # 10  # order book simulator levels
sim_queue_len = 100  # per price in sim, how many orders in queue

n_vol_series = 500  # how many book volume series model uses as input

# dataset_obj = LOBSTER(
#     'lobster',
#     # data_dir='/nfs/home/peern/LOBS5/data/new_enc/',
#     data_dir='/nfs/home/peern/LOBS5/data/fast_encoding/',
#     # use dummy mask function to get entire sequence
#     mask_fn=lambda X, rng: (X, jnp.array(0)),
#     use_book_data=True,
#     # book_transform=True,  # transform book to image
#     use_simple_book=True,  # return (p,v) book and we'll do transorm to volume image later
#     msg_seq_len=n_messages + n_eval_messages,
#     return_raw_msgs=True,
#     #raw_data_dir='/nfs/home/peern/LOBS5/data/raw/',
#     n_cache_files=100,  # keep high enough to fit all files in memory
# )
# dataset_obj.setup()

# ds = dataset_obj.dataset_test

msg_files = sorted(glob(str(data_dir) + '/*message*.npy'))
book_files = sorted(glob(str(data_dir) + '/*book*.npy'))

ds = LOBSTER_Dataset(
    msg_files,
    n_messages=n_messages + n_eval_messages,
    mask_fn=lambda X, rng: (X, jnp.array(0)),
    seed=42,
    n_cache_files=100,
    randomize_offset=False,
    book_files=book_files,
    use_simple_book=True,
    book_transform=False,
    book_depth=500,
    return_raw_msgs=True,
)

##################################################

import logging
# logging.basicConfig(filename='ar_debug.log', level=logging.DEBUG)
fhandler = logging.FileHandler(filename='ar_debug.log', mode='w')
logger = logging.getLogger()
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(fhandler)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.WARNING)

##################################################

results = inference.sample_messages(
    n_samples = 1000, # 500
    num_repeats = 1,
    ds = ds,
    rng = rng,
    seq_len = seq_len,
    n_msgs = n_messages,
    n_gen_msgs = n_gen_msgs,
    train_state = state,
    model = model,
    batchnorm = batchnorm,
    encoder = v.ENCODING,
    n_vol_series = n_vol_series,
    sim_book_levels = sim_book_levels,
    sim_queue_len = sim_queue_len,
    data_levels = data_levels,
    save_folder = save_dir
)
