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
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_depth: int = 500,
		n_data_workers: int = 0,
		return_raw_msgs: bool = False,
		horizon: int = None,
		horizon_type: str = None,
	) -> ReturnType:
	""" 
	"""

	print("[*] Generating LOBSTER Prediction Dataset from", cache_dir)
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	dataset_obj = LOBSTER(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		msg_seq_len=msg_seq_len,
		use_book_data=use_book_data,
		use_simple_book=use_simple_book,
		book_transform=book_transform,
		book_depth=book_depth,
		n_cache_files=1e7,  # large number to keep everything in cache
		return_raw_msgs=return_raw_msgs,
	)
	dataset_obj.setup()

	print("Using mask function:", mask_fn)

	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	trn_loader = create_lobster_train_loader(
		dataset_obj, seed, bsz, n_data_workers, reset_train_offsets=False)
	# NOTE: drop_last=True recompiles the model for a smaller batch size
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz,
		drop_last=True, shuffle=False, num_workers=n_data_workers)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz,
		drop_last=True, shuffle=False, num_workers=n_data_workers)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	BOOK_SEQ_LEN = dataset_obj.L_book
	BOOK_DIM = dataset_obj.d_book

	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders, 
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)


def create_FI_2010_classification_dataset(
		cache_dir: Union[str, Path] = DATA_DIR,
		seed: int = 42,
		mask_fn = None,
		msg_seq_len: int = 500,
		bsz: int=128,
		use_book_data: bool = False,
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_depth: int = 500,
		n_data_workers: int = 0,
		return_raw_msgs: bool = False,
		horizon: int = 10,
		horizon_type: str = 'messages',
	) -> ReturnType:
	""" 
	"""

	print("[*] Generating FI-2010 Prediction Dataset from", cache_dir)
	from .fi2010_dataloader import FI2010
	name = 'fi-2010'

	assert horizon_type == 'messages' , "For FI-2010 the horizon type must be event-based"



	dataset_obj = FI2010(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		input_length=msg_seq_len,
		use_book_data=use_book_data,
		use_simple_book=use_simple_book,
		book_transform=book_transform,
		book_depth=book_depth,
		n_cache_files=1e7,  # large number to keep everything in cache
		return_raw_msgs=return_raw_msgs,
		pred_horizon=horizon,
	)
	dataset_obj.setup()

	print("Using mask function:", mask_fn)

	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.datas-et_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	trn_loader = create_fi2010_train_loader(
		dataset_obj, seed, bsz, n_data_workers)
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz,
		shuffle=False, num_workers=n_data_workers)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz,
		shuffle=False, num_workers=n_data_workers)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	BOOK_SEQ_LEN = 0
	BOOK_DIM = 0


	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders, 
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)

def create_lobster_train_loader(dataset_obj, seed, bsz, num_workers, reset_train_offsets=False):
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()
	# use sampler to only get individual samples and automatic batching from dataloader
	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=bsz,
		shuffle=True,  # TODO: remove later
		num_workers=num_workers)
	return trn_loader

def create_fi2010_train_loader(dataset_obj, seed, bsz, num_workers):

	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=bsz,
		shuffle=False,
		num_workers=num_workers)
	return trn_loader

Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
	"FI-2010-classification": create_FI_2010_classification_dataset,
}
