from argparse import Namespace
from glob import glob
from functools import partial
from typing import Optional
import jax
import jax.numpy as np
from jax import random
from flax.training.train_state import TrainState
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
from orbax import checkpoint
from lob.encoding import Vocab
from lob.lob_seq_model import BatchFullLobPredModel, BatchLobPredModel

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state, eval_step, prep_batch, cross_entropy_loss, compute_accuracy
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.dataloading import make_data_loader
from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import validation_helpers as valh


def load_args_from_checkpoint(
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> Namespace:

    """Load arguments from checkpoint"""
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    raw_restored = checkpoints.restore_checkpoint(
        checkpoint_path,
        None,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    args = Namespace(**raw_restored['config'])
    return args


def load_checkpoint(
        state: TrainState,
        path: str,
        config_dict: dict,
        step: Optional[int] = None,
    ) -> TrainState:
    ckpt = {
        'model': state,
        'config': config_dict,
        'metrics': {
            'loss_train': np.nan,
            'loss_val': np.nan,
            'loss_test': np.nan,
            'acc_val': np.nan,
            'acc_test': np.nan,
        }
    }
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    restored = checkpoints.restore_checkpoint(
        path,
        ckpt,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    return restored


def init_train_state(
        args: Namespace,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        print_shapes=False
    ) -> TrainState:

    in_dim = n_classes

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)

    padded = False
    retrieval = False
    speech = False

    ssm_init_fn = init_S5SSM(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional
    )
    
    if args.use_book_data:
        model_cls = partial(
            BatchFullLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            d_book=book_dim,
            n_message_layers=2,  # TODO: make this an arg
            n_fused_layers=args.n_layers,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    else:
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global
    )

    return state
