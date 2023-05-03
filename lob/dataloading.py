import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from s5.dataloading import make_data_loader
from .lobster_dataloader import LOBSTER, LOBSTER_Dataset, LOBSTER_Sampler


DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
DATA_DIR = Path('../data/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[LOBSTER, DataLoader, DataLoader, DataLoader, Dict, int, int, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


def create_lobster_prediction_dataset(
		cache_dir: Union[str, Path] = DATA_DIR,
		seed: int = 42,
		mask_fn = LOBSTER_Dataset.causal_mask,
		msg_seq_len: int = 500,
		bsz: int=128,
		use_book_data: bool = False,
	) -> ReturnType:
	""" 
	"""

	print("[*] Generating LOBSTER Prediction Dataset from", cache_dir)
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	# kwargs = {
	# 	#'permute': True,
	# 	"mask_fn": mask_fn
	# }

	dataset_obj = LOBSTER(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		msg_seq_len=msg_seq_len,
		use_book_data=use_book_data,
	)
	dataset_obj.setup()

	print("Using mask function:", mask_fn)

	# TODO: make arg
	# TODO: same number of workers for val and test? (currently 0)
	num_workers = 4
	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	# trn_loader = make_data_loader(
	# 	dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz, sampler=trn_sampler, num_workers=num_workers)
	trn_loader = create_lobster_train_loader(dataset_obj, seed, bsz, num_workers, reset_train_offsets=False)
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False, num_workers=0)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False, num_workers=0)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	BOOK_SEQ_LEN = dataset_obj.L_book
	BOOK_DIM = dataset_obj.d_book

	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders, 
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)

def create_lobster_train_loader(dataset_obj, seed, bsz, num_workers, reset_train_offsets=False):
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()
	# use sampler to only get individual samples and automatic batching from dataloader
	trn_sampler = LOBSTER_Sampler(
		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=bsz,
		sampler=trn_sampler,
		num_workers=num_workers)
	return trn_loader

Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
}
