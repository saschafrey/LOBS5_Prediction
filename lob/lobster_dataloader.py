""" Datasets for core experimental results """
from pathlib import Path
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from glob import glob
import pandas as pd
import jax.numpy as jnp
from jax.nn import one_hot

from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.parent.absolute()
default_data_path = default_data_path / "data"


class LOBSTER_Dataset(Dataset):
    """ TODO: investigate speed of __getitem__ in practice: currently loads every sequence
              from pre-processed file, one-hot encodes and moves to GPU
        TODO: time encoding?
    """

    EVENT_TYPES = 2
    ORDER_SIZES = 64
    PRICES = 23

    def __init__(self, message_files, seq_len, n_buffer_files=0) -> None:
        assert len(message_files) > 0
        self.message_files = message_files #
        self.num_days = len(self.message_files)
        self.seq_len = seq_len
        # add prediction target to sequence
        self._seqs_per_file = np.array(
            #[self._get_num_rows(f) - (self.seq_len-1) for f in message_files])
            [self._get_num_rows(f) - (self.seq_len) for f in message_files])
        # store at which observations files start
        self._seqs_cumsum = np.concatenate(([0], np.cumsum(self._seqs_per_file)))
        # count total number of messages once
        self._len = int(self._seqs_cumsum[-1])
        # keep first "n_buffer_files" of accessed files in memory for faster access
        self.n_buffer_files = n_buffer_files
        self._file_buffer = dict()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        file_idx, seq_idx = self._get_seq_location(idx)
        # load file in buffer if possible
        self._try_load_buffer(file_idx)
            
        # access data from buffer
        if file_idx in self._file_buffer:
            X = self._file_buffer[file_idx][seq_idx: seq_idx + self.seq_len + 1]
        # load sequence from file directly without buffer
        else:
            #print('getting from file', file_idx, 'item', seq_idx)
            df = pd.read_csv(
                self.message_files[file_idx],
                names = ['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
                index_col = False,
                skiprows=seq_idx,
                nrows=self.seq_len + 1
            )
            X = df[['time', 'event_type', 'size', 'price', 'direction']].values
            X = torch.tensor(self._encode_features(X))
        
        # TODO: encode and add time
        # X, y (last element of sequence), aux_data (time data)
        return X[:-1, 1:], X[-1, 1:], X[:, 0]

    def _try_load_buffer(self, file_idx):
        """ add file to buffer if not full and not yet in buffer
        """
        if (len(self._file_buffer) < self.n_buffer_files) and (file_idx not in self._file_buffer):
            df = pd.read_csv(
                self.message_files[file_idx],
                names = ['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
                index_col = False
            )
            X = df[['time', 'event_type', 'size', 'price', 'direction']].values
            X = torch.tensor(self._encode_features(X))
            self._file_buffer[file_idx] = X
    
    def _encode_features(self, X):
        return np.concatenate(
            (
                np.array(X[:, 0].reshape((-1,1))),  # leave time as is for now
                one_hot(X[:, 1], LOBSTER_Dataset.EVENT_TYPES),
                one_hot(X[:, 2], LOBSTER_Dataset.ORDER_SIZES),
                one_hot(X[:, 3], LOBSTER_Dataset.PRICES),
                one_hot(X[:, 4], 2),  # encode direction as two dummy cols as well
                #np.array(X[:, 4].reshape((-1,1)))  # direction is already in {0,1}
            ),
            axis=1)

    def _get_num_rows(self, file_path):
        with open(file_path) as f:
            return sum(1 for line in f)

    def _get_seq_location(self, idx):
        file_idx = np.searchsorted(self._seqs_cumsum, idx+1) - 1
        seq_idx = idx - self._seqs_cumsum[file_idx]
        return file_idx, seq_idx


class LOBSTER(SequenceDataset):
    _name_ = "lobster"
    d_input = (
        LOBSTER_Dataset.EVENT_TYPES,
        LOBSTER_Dataset.ORDER_SIZES,
        LOBSTER_Dataset.PRICES,
        2)  # direction
    d_output = d_input
    l_output = 0
    L = 500

    _collate_arg_names = ['timesteps']

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        """
        Custom collate function.
        Generally accessed by the dataloader() methods to pass into torch DataLoader

        Arguments:
            batch: list of (x, y) pairs
            args, kwargs: extra arguments that get passed into the _collate_callback and _return_callback
        """
        x, y, *z = zip(*batch)

        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]

        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "k_val_segments": 5,  # train/val split is done by picking 5 continguous folds
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = default_data_path
        message_files = sorted(glob(str(self.data_dir) + '/*message*'))
        n_test_files = max(1, int(len(message_files) * self.test_split))

        # train on first part of data
        train_files = message_files[:len(message_files) - n_test_files]
        # and test on last days
        test_files = message_files[len(train_files):]
        
        # split into train/val in split_train_val()
        self.dataset_train = LOBSTER_Dataset(
            train_files,
            seq_len=self.L,
            n_buffer_files=5
        )
        self.split_train_val(self.val_split)

        self.dataset_test = LOBSTER_Dataset(
            test_files,
            seq_len=self.L,
            n_buffer_files=2
        )
        # TODO: remove
        # decrease test size to run faster:
        self.dataset_test = Subset(self.dataset_test, range(int(0.1 * len(self.dataset_test))))
        

    def split_train_val(self, val_split):
        """ splits current dataset_train into separate dataset_train and dataset_val
            by selecting k_val_segments contiguous folds from the sequence dataset.
            Sequences with overlapping observations are removed from dataset_train.
        """
        #train_len = int(len(self.dataset_train) * (1.0 - val_split))
        indices = np.arange(len(self.dataset_train))
        
        n_segments = int(self.k_val_segments / val_split)
        folds = np.array_split(indices, n_segments)
        random.seed(getattr(self, "seed", 42))
        val_segments = random.sample(range(len(folds)), self.k_val_segments)
        train_indices = np.hstack(tuple(folds[i] for i in range(n_segments) if i not in val_segments))

        val_indices = np.hstack(tuple(folds[i] for i in val_segments))
        # remove from training data, those sequences that overlap with validation
        remove_train = set.union(*[{i+j for j in range(-self.L+1, self.L)} for i in val_indices])
        train_indices = list(set(train_indices) - remove_train)

        #print('train:', len(train_indices))
        #print(train_indices[:10])
        #print('val:', len(val_indices))
        #print(val_indices[:10])

        self.dataset_val = Subset(self.dataset_train, val_indices)
        self.dataset_train = Subset(self.dataset_train, train_indices)
        
    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"
