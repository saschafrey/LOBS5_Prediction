import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from s5.dataloading import make_data_loader
from .lobster_dataloader import LOBSTER, LOBSTER_Dataset


DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
DATA_DIR = Path('../data/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


def create_lobster_prediction_dataset(cache_dir: Union[str, Path] = DATA_DIR,
									  seed: int = 42,
									  mask_fn = LOBSTER_Dataset.causal_mask,
									  bsz: int=128) -> ReturnType:
	""" 
	"""

	print("[*] Generating LOBSTER Prediction Dataset")
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	kwargs = {
		'permute': True,
		"mask_fn": mask_fn
	}

	dataset_obj = LOBSTER(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE

Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
}
