""" Datasets for core experimental results """
from pathlib import Path
import random
import sys
from typing import Sequence
import numpy as np
from collections import OrderedDict
import torch
import torchvision
from torch.utils.data import Dataset, Subset, Sampler
from glob import glob
import pandas as pd
import jax.numpy as jnp
from jax.nn import one_hot
from jax.experimental import sparse

from lob.encoding import Vocab, Message_Tokenizer
from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.parent.absolute()
default_data_path = default_data_path / "data"


class LOBSTER_Dataset(Dataset):
    """ TODO: investigate speed of __getitem__ in practice: currently loads every sequence
              from pre-processed file, one-hot encodes and moves to GPU
        TODO: time encoding?
    """

    #EVENT_TYPES = 2
    #ORDER_SIZES = 64
    #PRICES = 23

    # TODO: random seed for mask positioning?
    @staticmethod
    def random_mask(seq, rng):
        """ Select random token in given seq and set to MSK token
            as prediction target
        """
        i = rng.integers(0, len(seq.flat) - 1)
        y = seq.flat[i]
        seq.flat[i] = Vocab.MASK_TOK
        return seq, y

    @staticmethod
    def causal_mask(seq, rng):
        """ Select random field (e.g price) in most recent message
            for which one token is MSKd (tokens left of MSK are know,
            right of MSK are NA). MSK token becomes prediction target.
            Random subset of other fields are also set to NA.
            This simulates the causal prediction task, where fields
            can be predicted in arbitrary order.
        """

        hidden_fields, msk_field = LOBSTER_Dataset._select_random_causal_mask(rng)
        i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)
        msk_i = rng.integers(i_start, i_end)
        # select random token from last message from selected field
        y = seq[-1][msk_i]
        seq[-1][msk_i] = Vocab.MASK_TOK
        # set tokens after MSK token to HIDDEN for masked field
        if msk_i < (i_end - 1):
            seq[-1][msk_i + 1: i_end] = Vocab.HIDDEN_TOK
        # set all hidden_fields to HIDDEN
        for f in hidden_fields:
            seq[-1][slice(*LOBSTER_Dataset._get_tok_slice_i(f))] = Vocab.HIDDEN_TOK
        return seq, y

    @staticmethod
    def _select_random_causal_mask(rng):
        n_fields = len(Message_Tokenizer.TOK_LENS)
        sel_fields = sorted(rng.choice(
            list(range(n_fields)),
            rng.integers(1, n_fields + 1), replace=False
        ))
        msk_field = rng.choice(sel_fields)
        sel_fields = list(set(sel_fields) - {msk_field})
        return sel_fields, msk_field

    @staticmethod
    def _get_tok_slice_i(field_i):
        i_start = ([0] + list(Message_Tokenizer.TOK_DELIM))[field_i]
        field_len = Message_Tokenizer.TOK_LENS[field_i]
        return i_start, i_start + field_len

    def __init__(
            self,
            message_files,
            n_messages,
            mask_fn,
            seed=None,
            n_buffer_files=0,
            randomize_offset=True,
            ) -> None:

        assert len(message_files) > 0
        self.message_files = message_files #
        self.num_days = len(self.message_files)
        self.n_messages = n_messages
        
        # keep first "n_buffer_files" of accessed files in memory for faster access
        self.n_buffer_files = n_buffer_files
        self._file_cache = OrderedDict()
        self.vocab = Vocab()
        self.seq_len = self.n_messages * Message_Tokenizer.MSG_LEN
        self.mask_fn = mask_fn
        self.rng = np.random.default_rng(seed)
        self.randomize_offset = randomize_offset
        self._reset_offsets()

        #self._seqs_per_file = np.array(
        #    [self._get_num_rows(f) - (self.n_messages-1) for f in message_files])
        self._seqs_per_file = np.array(
            [(self._get_num_rows(f) - self.seq_offsets[i]) // n_messages
             for i, f in enumerate(message_files)])
        # store at which observations files start
        self._seqs_cumsum = np.concatenate(([0], np.cumsum(self._seqs_per_file)))
        # count total number of sequences only once
        self._len = int(self._seqs_cumsum[-1])

    def _reset_offsets(self):
        """ drop a random number of messages from the beggining of every file
            so that sequences don't always contain the same time periods
        """
        if self.randomize_offset:
            self.seq_offsets = {
                i: self.rng.integers(0, self.n_messages) 
                for i in range(len(self.message_files))}
        else:
            self.seq_offsets = {i: 0 for i in range(len(self.message_files))}

    @property
    def shape(self):
        return len(self), Message_Tokenizer.MSG_LEN#, len(self.vocab)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        file_idx, seq_idx = self._get_seq_location(idx)
        
        # load sequence from file directly without cache
        if self.n_buffer_files == 0:
            X = np.load(self.message_files[file_idx], mmap_mode='r')
        else:
            if file_idx not in self._file_cache:
                self._add_to_cache(file_idx)

            X = self._file_cache[file_idx]

        seq_start = self.seq_offsets[file_idx] + seq_idx * self.n_messages
        seq_end = seq_start + self.n_messages
        #print('len X', len(X))
        #print(f'slice ({seq_start}, {seq_end})')
        X = X[seq_start: seq_end]
        # apply mask and extract prediction target token
        X, y = self.mask_fn(X, self.rng)
        X, y = X.reshape(-1), y.reshape(-1)
        # TODO: look into aux_data (could we still use time when available?)
        return X, y#, None

    def _add_to_cache(self, file_idx):
        if len(self._file_cache) >= self.n_buffer_files:
            # remove item in FIFO order
            _ = self._file_cache.popitem(last=False)
            del _
        X = np.load(self.message_files[file_idx])
        self._file_cache[file_idx] = X

    '''
    def __getitem__(self, idx):
        file_idx, seq_idx = self._get_seq_location(idx)
        # load file in cache if possible
        self._try_load_cache(file_idx)
            
        # access data from cache
        if file_idx in self._file_cache:
            X = self._file_cache[file_idx]
        # load sequence from file directly without cache
        else:
            X = np.load(self.message_files[file_idx], mmap_mode='r')

        X = X[seq_idx: seq_idx + self.n_messages]
        #print('pre mask fn', X.shape)

        # apply mask and extract prediction target token
        X, y = self.mask_fn(X, self.rng)

        #print('pre enc', X.shape, y.shape)

        # onehot encode
        #X = self._encode_features(X.reshape(-1))
        X = X.reshape(-1)
        #y = self._encode_features(y.reshape(-1))
        y = y.reshape(-1)
        #print('post enc', X.shape, y.shape)
        
        # X, y (last element of sequence), aux_data (time data)
        #return X[:-1, :], X[-1, :], X[:, 0]
        # TODO: look into aux_data (could we still use time when available?)
        return X, y#, None

    def _try_load_cache(self, file_idx):
        """ add file to cache if not full and not yet in cache
            TODO: implement proper cache
        """
        if (len(self._file_cache) < self.n_buffer_files) and (file_idx not in self._file_cache):
            #df = pd.read_csv(
            #    self.message_files[file_idx],
            #    names = ['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
            #    index_col = False
            #)
            #X = df[['time', 'event_type', 'size', 'price', 'direction']].values
            X = np.load(self.message_files[file_idx])
            #X = torch.tensor(self._encode_features(X))
            #X = self._encode_features(X)
            self._file_cache[file_idx] = X
    '''
    
    def _encode_features(self, X):
        """ DEPRECATED: one-hot encoding done for entire batch instead
        """
        vocab_size = len(self.vocab)
        res = np.eye(vocab_size)[X.reshape(-1)]
        res = res.reshape(list(X.shape)+[vocab_size])
        return res

    def _get_num_rows(self, file_path):
        #with open(file_path) as f:
        #    return sum(1 for line in f)
        # only load data header and return length
        d = np.load(file_path, mmap_mode='r', allow_pickle=True)
        return d.shape[0]

    def _get_seq_location(self, idx):
        if idx > len(self) - 1:
            raise IndexError(f'index {idx} out of range for dataset length ({len(self)})')
        file_idx = np.searchsorted(self._seqs_cumsum, idx+1) - 1
        seq_idx = idx - self._seqs_cumsum[file_idx]
        #seq_idx = idx - self._seqs_cumsum[file_idx] + self.seq_offsets[file_idx]
        return file_idx, seq_idx
    

class LOBSTER_Sampler(Sampler):
    def __init__(self, dset, n_files_shuffle, batch_size=1, seed=None):
        self.dset = dset
        self.n_files_shuffle = n_files_shuffle
        self.batch_size = batch_size

        self.rng = random.Random(seed)

        # LOBSTER_Dataset
        if hasattr(self.dset, "num_days"):
            days = range(dset.num_days)
        # LOBSTER_Subset
        elif hasattr(self.dset, "indices_on_day"):
            days = list(self.dset.indices_on_day.keys())
        else:
            raise AttributeError("dataset has neither num_days nor indices_on_day attribute.")
        # days in random order
        self.days_unused = self.rng.sample(
            days,
            len(days)
        )
        self.active_indices = []

    def __iter__(self):
        while len(self.days_unused) > 0 or len(self.active_indices) >= self.batch_size:
            batch = []
            # not enough indices available for full batch
            if len(self.active_indices) < self.batch_size:
                batch += list(self.active_indices)
                # get new indices from new days
                self.active_indices = self._get_new_active_indices(self._get_new_days())
            while len(batch) < self.batch_size:
                # TODO: fix pop from empty list error
                batch.append(self.active_indices.pop())
            if self.batch_size == 1:
                batch = batch[0]
            yield batch

    def _get_new_days(self):
        days = []
        for _ in range(self.n_files_shuffle):
            if len(self.days_unused) > 0:
                days.append(self.days_unused.pop())
            else:
                break
        return days

    def _get_new_active_indices(self, days):
        idx = []
        # LOBSTER_Dataset
        if hasattr(self.dset, "_seqs_cumsum"):
            for d in days:
                idx.extend(
                    list(range(
                        self.dset._seqs_cumsum[d],
                        self.dset._seqs_cumsum[d + 1]
                    ))
                )
        elif hasattr(self.dset, "indices_on_day"):
            for d in days:
                idx.extend(self.dset.indices_on_day[d])
        else:
            raise AttributeError("dataset has neither num_days nor indices_on_day attribute.")
        
        self.rng.shuffle(idx)
        return idx

    def __len__(self):
        return len(self.dset)


class LOBSTER_Subset(Subset):
    def __init__(self, dataset: LOBSTER_Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = sorted(indices)

        self.indices_on_day = self.get_indices_by_day(
            self.indices)

    def get_indices_by_day(self, indices):
        indices_on_day = {}
        day = 0
        i_end = self.dataset._seqs_cumsum[day + 1]

        for idx in indices:
            while idx >= i_end:
                day += 1
                i_end = self.dataset._seqs_cumsum[day + 1]
            
            if day not in indices_on_day.keys():
                indices_on_day[day] = []
            indices_on_day[day].append(idx)
        return indices_on_day


class LOBSTER(SequenceDataset):
    _name_ = "lobster"
    #d_input = (
    #    LOBSTER_Dataset.EVENT_TYPES,
    #    LOBSTER_Dataset.ORDER_SIZES,
    #    LOBSTER_Dataset.PRICES,
    #    2)  # direction
    #d_output = d_input
    l_output = 0
    n_messages = 500
    #L = 500

    _collate_arg_names = [] #['timesteps']

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
            "permute": True,  # TODO: implement efficient permutation (within buffered files)
            "k_val_segments": 5,  # train/val split is done by picking 5 contiguous folds
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # For train/val split
            "mask_fn": LOBSTER_Dataset.random_mask
        }

    def setup(self):
        self.data_dir = default_data_path
        message_files = sorted(glob(str(self.data_dir) + '/*message*.npy'))
        n_test_files = max(1, int(len(message_files) * self.test_split))

        # train on first part of data
        train_files = message_files[:len(message_files) - n_test_files]
        # and test on last days
        test_files = message_files[len(train_files):]

        self.rng = random.Random(self.seed)

        # split into train/val in split_train_val()
        # TODO: parametrise task as either random tokenization vs causal?
        self.dataset_train = LOBSTER_Dataset(
            train_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_buffer_files=5,
            randomize_offset=True,
        )
        self.d_input = self.dataset_train.shape[-1]
        self.d_output = self.d_input
        # sequence length
        self.L = self.n_messages * Message_Tokenizer.MSG_LEN

        self.split_train_val(self.val_split)

        self.dataset_test = LOBSTER_Dataset(
            test_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_buffer_files=2,
            randomize_offset=False,
        )

        # TODO: remove
        # decrease test size for now to run faster:
        self.dataset_test = LOBSTER_Subset(self.dataset_test, range(int(0.1 * len(self.dataset_test))))

    def split_train_val(self, val_split):
        """ takes a random subset of training data as validation data
        """
        rng = random.Random(self.seed)
        indices = list(range(len(self.dataset_train))) #np.arange(len(self.dataset_train))
        val_len = int(np.ceil(val_split * len(indices)))
        val_indices = [indices.pop(rng.randrange(0, len(indices))) for _ in range(val_len)]

        self.dataset_val = LOBSTER_Subset(self.dataset_train, val_indices)
        self.dataset_train = LOBSTER_Subset(self.dataset_train, indices)

    '''
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
        remove_train = set.union(*[{i+j for j in range(-self.n_messages+1, self.n_messages)} for i in val_indices])
        train_indices = list(set(train_indices) - remove_train)

        #print('train:', len(train_indices))
        #print(train_indices[:10])
        #print('val:', len(val_indices))
        #print(val_indices[:10])

        self.dataset_val = Subset(self.dataset_train, val_indices)
        self.dataset_train = Subset(self.dataset_train, train_indices)
    '''

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"
