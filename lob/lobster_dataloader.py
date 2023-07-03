""" Datasets for core experimental results """
from pathlib import Path
import random
import sys
from typing import Sequence
import numpy as np
from collections import OrderedDict

#import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

#import torch
#import torchvision
from torch.utils.data import Dataset, Subset, Sampler
from glob import glob
import pandas as pd
import jax
# Global flag to set a specific platform, must be used at startup.
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.nn import one_hot
# from jax.experimental import sparse

import lob.encoding as encoding
from lob.encoding import Vocab, Message_Tokenizer
from lob.preproc import transform_L2_state
from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.parent.absolute()
default_data_path = default_data_path / "data"


class LOBSTER_Dataset(Dataset):

    @staticmethod
    def get_masking_fn(*, random_msg_idxs=None, random_fields=None, randomize_message=True):
        """ Get masking function for given fields
            random_msg_idxs: list of message indices to mask
            random_fields: list of fields to mask.
                           NOTE: either this or random_msg_idxs must be given
            randomize_message: if True, select random message to mask, otherwise
                               mask most recent message
        """
        assert (random_msg_idxs is None) != (random_fields is None)
        if random_fields is not None:
            # get corresponding field indices
            random_fields = np.array(random_fields)
            ref = np.array(Message_Tokenizer.FIELDS)
            field_idxs = np.array([np.argwhere(f==ref) for f in random_fields]).flatten()
            random_msg_idxs = [
                idx 
                for f in field_idxs
                for idx in range(*LOBSTER_Dataset._get_tok_slice_i(f))]
        
        def masking_fn(seq, rng):
            seq = seq.copy()
            if randomize_message:
                m_i = rng.integers(0, seq.shape[0])
            else:
                m_i = seq.shape[0] - 1
            if len(random_msg_idxs) == 1:
                tok_i = random_msg_idxs[0]
            else:
                tok_i = rng.choice(random_msg_idxs)
            y = seq[m_i, tok_i]
            seq[m_i, tok_i] = Vocab.MASK_TOK
            return seq, y
            
        return masking_fn

    # @staticmethod
    # def random_mask_all(seq, rng):
    #     """ mask a random token somewhere in the sequence """
    #     seq = seq.copy()
    #     i = rng.integers(0, len(seq.flat) - 1)
    #     y = seq.flat[i]
    #     seq.flat[i] = Vocab.MASK_TOK
    #     return seq, y

    @staticmethod
    def random_mask(seq, rng, exclude_time=True):
        """ Select random token in given seq and set to MSK token
            as prediction target
        """
        # mask a random token in the most recent message
        # and HIDe a random uniform number of other tokens randomly
        seq = seq.copy()

        # select random positions to HIDe
        l = Message_Tokenizer.MSG_LEN

        # exclude time from masking if exclude_time == True
        if exclude_time:
            time_field_i = Message_Tokenizer.FIELD_I['time']
            time_start_i, time_end_i = LOBSTER_Dataset._get_tok_slice_i(time_field_i)
            candidate_pos = list(range(time_start_i)) + list(range(time_end_i, l))
            max_hid = l + 1 - (time_end_i - time_start_i)
        else:
            candidate_pos = list(range(l))
            max_hid = l + 1

        # sample uniformly without replacement
        hid_pos = sorted(
            rng.choice(
                #list(range(l)),
                candidate_pos,
                rng.integers(1, max_hid),
                replace=False
            )
        )

        # select one position to MSK from pre-selected HID positions
        msk_pos = rng.choice(hid_pos)
        hid_pos.remove(msk_pos)

        # deterministically hide time if delta_t is not complete (some HID)
        if exclude_time:
            dt_field_i = Message_Tokenizer.FIELDS.index('delta_t')
            dt_idx = list(range(*LOBSTER_Dataset._get_tok_slice_i(dt_field_i)))
            # part of delta_t is hidden
            if any(i in hid_pos for i in dt_idx):
                # --> also HIDe time
                hid_pos.extend(range(time_start_i, time_end_i))

        y = seq[-1, msk_pos]
        seq[-1, msk_pos] = Vocab.MASK_TOK
        seq[-1, hid_pos] = Vocab.HIDDEN_TOK
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
        seq = seq.copy()
        #hidden_fields, msk_field = LOBSTER_Dataset._select_random_causal_mask(rng)
        #hidden_fields, msk_field = LOBSTER_Dataset._select_unconditional_mask(rng)
        # hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask(rng)
        hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask_no_time(rng)

        i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)
        msk_i = rng.integers(i_start, i_end)
        # select random token from last message from selected field
        y = seq[-1][msk_i]
        # seq[-1][msk_i] = Vocab.MASK_TOK
        seq = seq.at[-1, msk_i].set(Vocab.MASK_TOK)
        # set tokens after MSK token to HIDDEN for masked field
        if msk_i < (i_end - 1):
            # seq[-1][msk_i + 1: i_end] = Vocab.HIDDEN_TOK
            seq = seq.at[-1, msk_i + 1: i_end].set(Vocab.HIDDEN_TOK)
        # set all hidden_fields to HIDDEN
        for f in hidden_fields:
            # seq[-1][slice(*LOBSTER_Dataset._get_tok_slice_i(f))] = Vocab.HIDDEN_TOK
            seq = seq.at[-1, slice(*LOBSTER_Dataset._get_tok_slice_i(f))].set(Vocab.HIDDEN_TOK)
        return seq, y
    
    #@jax.jit
    # @staticmethod
    # def causal_mask(seq, rng):
    #     """ Select random field (e.g price) in most recent message
    #         for which one token is MSKd (tokens left of MSK are know,
    #         right of MSK are NA). MSK token becomes prediction target.
    #         Random subset of other fields are also set to NA.
    #         This simulates the causal prediction task, where fields
    #         can be predicted in arbitrary order.
    #     """
    #     def get_tok_slice_i(field_i):
    #         i_start = ([0] + list(Message_Tokenizer.TOK_DELIM))[field_i]
    #         field_len = Message_Tokenizer.TOK_LENS[field_i]
    #         return i_start, i_start + field_len
        
    #     seq = seq.copy()
    #     #hidden_fields, msk_field = LOBSTER_Dataset._select_random_causal_mask(rng)
    #     #hidden_fields, msk_field = LOBSTER_Dataset._select_unconditional_mask(rng)
    #     hidden_fields, msk_field = LOBSTER_Dataset._select_sequential_causal_mask(rng)

    #     i_start, i_end = LOBSTER_Dataset._get_tok_slice_i(msk_field)
    #     msk_i = rng.integers(i_start, i_end)
    #     #msk_i = jax.random.randint(rng, (1,), minval=i_start, maxval=i_end)
    #     # select random token from last message from selected field
    #     y = seq[-1][msk_i]
    #     seq = seq.at[-1, msk_i].set(Vocab.MASK_TOK)
    #     # set tokens after MSK token to HIDDEN for masked field
    #     if msk_i < (i_end - 1):
    #         seq = seq.at[-1, msk_i + 1: i_end].set(Vocab.HIDDEN_TOK)
    #     # set all hidden_fields to HIDDEN
    #     for f in hidden_fields:
    #         seq = seq.at[-1, slice(*get_tok_slice_i(f))].set(
    #             Vocab.HIDDEN_TOK)
    #     return seq, y

    @staticmethod
    def _select_random_causal_mask(rng):
        """ Select random subset of fields and one field to mask
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        sel_fields = sorted(rng.choice(
            list(range(n_fields)),
            rng.integers(1, n_fields + 1),
            replace=False
        ))

        msk_field = rng.choice(sel_fields)
        sel_fields = list(set(sel_fields) - {msk_field})
        return sel_fields, msk_field

    @staticmethod
    def _select_unconditional_mask(rng):
        """ Select only one field to mask and all other fields to be hidden
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        sel_fields = list(range(n_fields))
        msk_field = sel_fields.pop(rng.integers(0, n_fields))
        return sel_fields, msk_field
    
    @staticmethod
    def _select_sequential_causal_mask(rng):
        """ Select tokens random field till end for HID
            preceded by one MSK
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        msk_field = rng.integers(0, n_fields)
        #return tuple(range(msk_field)), msk_field
        return tuple(range(msk_field + 1, n_fields)), msk_field

    @staticmethod
    def _select_sequential_causal_mask_no_time(rng):
        """ Select tokens from random field to end for HID
            preceded by one MSK
            TIME field is never MSKd (not predicted)
        """
        n_fields = len(Message_Tokenizer.FIELDS)
        msk_field = rng.integers(0, n_fields - 1)
        i_time_s = Message_Tokenizer.FIELDS.index('time_s')
        i_time_ns = Message_Tokenizer.FIELDS.index('time_ns')
        if msk_field == i_time_s:
            msk_field += 2
        elif msk_field >= i_time_ns:
            msk_field += 1
        # hidden_fields, msk_field
        return tuple(range(msk_field + 1, n_fields)), msk_field

    # @staticmethod
    # def _select_sequential_causal_mask(rng):
    #     """ Select tokens from start to random field for HID
    #         followed by one MSK
    #     """
    #     n_fields = len(Message_Tokenizer.FIELDS)
    #     msk_field = jax.random.randint(rng, (1,), minval=0, maxval=n_fields)
    #     return jnp.arange(msk_field + 1, n_fields, dtype=jnp.int32), msk_field

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
            n_cache_files=0,
            randomize_offset=True,
            *,
            book_files=None,
            use_simple_book=False,
            book_transform=True,
            book_depth=500,
            # if given, also load and return raw sequences
            # -> used for inference (not training!)
            return_raw_msgs=False,
            ) -> None:

        assert len(message_files) > 0
        assert not (use_simple_book and book_transform)

        self.message_files = message_files #
        if book_files is not None:
            assert len(book_files) == len(message_files)
            self.use_book_data = True
            self.book_files = book_files
            self._book_cache = OrderedDict()
        else:
            self.use_book_data = False
        self.use_simple_book = use_simple_book
        self.book_transform = book_transform
        self.book_depth = book_depth
        self.return_raw_msgs = return_raw_msgs
        self.num_days = len(self.message_files)
        self.n_messages = n_messages

        self.n_cache_files = n_cache_files
        self._message_cache = OrderedDict()
        self.vocab = Vocab()
        self.seq_len = self.n_messages * Message_Tokenizer.MSG_LEN
        self.mask_fn = mask_fn
        self.rng = np.random.default_rng(seed)
        self.rng_jax = jax.random.PRNGKey(seed)
        self.randomize_offset = randomize_offset
        self._reset_offsets()
        self._set_book_dims()

        #self._seqs_per_file = np.array(
        #    [self._get_num_rows(f) - (self.n_messages-1) for f in message_files])
        self._seqs_per_file = np.array(
            [(self._get_num_rows(f) - self.seq_offsets[i]) // n_messages
             for i, f in enumerate(message_files)])
        # store at which observations files start
        self._seqs_cumsum = np.concatenate(([0], np.cumsum(self._seqs_per_file)))
        # count total number of sequences only once
        self._len = int(self._seqs_cumsum[-1])

    def _set_book_dims(self):
        if self.use_book_data:
            if self.book_transform:
                self.d_book = self.book_depth + 1
            else:
                b = np.load(self.book_files[0], mmap_mode='r', allow_pickle=True)
                self.d_book = b.shape[1]
            # TODO: generalize to L3 data
            self.L_book = self.n_messages
        else:
            self.d_book = 0
            self.L_book = 0
    
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
        if hasattr(idx, '__len__'):
            return list(zip(*[self[i] for i in idx]))

        file_idx, seq_idx = self._get_seq_location(idx)
        
        # load sequence from file directly without cache
        if self.n_cache_files == 0:
            X = np.load(self.message_files[file_idx], mmap_mode='r')
            if self.use_book_data:
                book = np.load(self.book_files[file_idx], mmap_mode='r')
        else:
            if file_idx not in self._message_cache:
                self._add_to_cache(file_idx)

            #print('fetching from cache')
            X = self._message_cache[file_idx]
            if self.use_book_data:
                book = self._book_cache[file_idx]

        seq_start = self.seq_offsets[file_idx] + seq_idx * self.n_messages
        seq_end = seq_start + self.n_messages
        
        X_raw = jnp.array(X[seq_start: seq_end])
        # encode message
        
        # if self.return_raw_msgs:
        #     X = encoding.encode_msgs(X_raw, self.vocab.ENCODING)
        # else:
        #     X = X_raw
        X = encoding.encode_msgs(X_raw, self.vocab.ENCODING)

        # apply mask and extract prediction target token
        X, y = self.mask_fn(X, self.rng)
        #X, y = self.mask_fn(X, self.rng_jax)
        X, y = X.reshape(-1), y.reshape(-1)
        # TODO: look into aux_data (could we still use time when available?)

        if self.use_book_data:
            # first message is already dropped, so we can use
            # the book state with the same index (prior to the message)
            book = book[seq_start: seq_end].copy()#.reshape(-1)
            if self.return_raw_msgs:
                book_l2_init = book[0, 1:].copy()

            # tranform from L2 (price volume) representation to fixed volume image 
            if self.book_transform:
                #book = jax.device_put(book, device=jax.devices("cpu")[0])
                #book = np.array(transform_L2_state(book, self.book_depth, 100))
                book = transform_L2_state(book, self.book_depth, 100)

            # use raw price, volume series, rather than volume image
            # subtract initial price to start all sequences around 0
            if self.use_simple_book:
                # CAVE: first column is Delta mid price
                # p_mid_0 = (book[0, 1] + book[0, 3]) / 2
                # book[:, 1::2] = (book[:, 1::2] - p_mid_0)
                # divide volume by 100
                #book[:, 2::2] = book[:, 2::2] / 100
                pass

            ret_tuple = X, y, book
        else:
            ret_tuple = X, y

        if self.return_raw_msgs:
            if self.use_book_data:
                ret_tuple += (X_raw, book_l2_init)
            else:
                ret_tuple += (X_raw,)

        return ret_tuple

    def _add_to_cache(self, file_idx):
        if len(self._message_cache) >= self.n_cache_files:
            # remove item in FIFO order
            _ = self._message_cache.popitem(last=False)
            if self.use_book_data:
                _ = self._book_cache.popitem(last=False)
            del _

        Xm = np.load(self.message_files[file_idx], mmap_mode='r')
        self._message_cache[file_idx] = Xm
        
        if self.use_book_data:
            Xb = np.load(self.book_files[file_idx], mmap_mode='r')
            self._book_cache[file_idx] = Xb

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
        assert n_files_shuffle > 0
        self.n_files_shuffle = n_files_shuffle
        self.batch_size = batch_size

        self.rng = random.Random(seed)

    def reset(self):
        # LOBSTER_Dataset
        if hasattr(self.dset, "num_days"):
            days = range(self.dset.num_days)
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
        # reset days and active indices whenever new iterator is created (e.g. new epoch)
        self.reset()

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

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def get_indices_by_day(self, indices):
        indices_on_day = {}
        day = 0
        i_end = self.dataset._seqs_cumsum[day + 1]

        for i, idx in enumerate(indices):
            while idx >= i_end:
                day += 1
                i_end = self.dataset._seqs_cumsum[day + 1]
            
            if day not in indices_on_day.keys():
                indices_on_day[day] = []
            indices_on_day[day].append(i)
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
    #n_messages = 500

    _collate_arg_names = ['book_data'] #['book_data'] #['timesteps']

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
        # NOTE: don't add data_dir here, it's added in the base class
        return {
            #"permute": False,
            #"k_val_segments": 5,  # train/val split is done by picking 5 contiguous folds
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # For train/val split
            "mask_fn": LOBSTER_Dataset.random_mask,
            "use_book_data": False,
            "use_simple_book" : False,
            "book_transform": False,
            "n_cache_files": 0,
            "book_depth": 500,
            "return_raw_msgs": False,
        }

    def setup(self):
        self.n_messages = self.msg_seq_len
        #self.data_dir = default_data_path
        message_files = sorted(glob(str(self.data_dir) + '/*message*.npy'))
        assert len(message_files) > 0, f'no message files found in {self.data_dir}'
        if self.use_book_data:
            # TODO: why does this only work for validation?
            #       can this be variable depending on the dataset?
            #self._collate_arg_names.append('book_data')
            book_files = sorted(glob(str(self.data_dir) + '/*book*.npy'))
            assert len(message_files) == len(book_files)
        else:
            book_files = None
        # raw message files

        n_test_files = max(1, int(len(message_files) * self.test_split))

        # train on first part of data
        self.train_files = message_files[:len(message_files) - n_test_files]
        # and test on last days
        self.test_files = message_files[len(self.train_files):]

        self.rng = random.Random(self.seed)

        # TODO: case of raw data but no book data?

        if book_files:
            self.train_book_files = book_files[:len(book_files) - n_test_files]
            self.test_book_files = book_files[len(self.train_book_files):]
            # zip together message and book files to randomly sample together
            self.train_files = list(zip(self.train_files, self.train_book_files))
        else:
            self.train_book_files = None
            self.val_book_files = None
            self.test_book_files = None

        # for now, just select (e.g. 10% of) days randomly for validation
        self.val_files = [
            self.train_files.pop(
                self.rng.randrange(0, len(self.train_files))
            ) for _ in range(int(np.ceil(self.val_split * len(message_files))))]
        if book_files:
            self.train_files, self.train_book_files = zip(*self.train_files)
            self.val_files, self.val_book_files = zip(*self.val_files)

        #n_cache_files = 0
        self.dataset_train = LOBSTER_Dataset(
            self.train_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_cache_files=self.n_cache_files,
            randomize_offset=True,
            book_files=self.train_book_files,
            use_simple_book=self.use_simple_book,
            book_transform=self.book_transform,
            book_depth=self.book_depth,
            return_raw_msgs=self.return_raw_msgs,
        )
        #self.d_input = self.dataset_train.shape[-1]
        self.d_input = len(self.dataset_train.vocab)
        self.d_output = self.d_input
        # sequence length
        self.L = self.n_messages * Message_Tokenizer.MSG_LEN
        # book sequence lengths and dimension (number of levels + 1)
        self.L_book = self.dataset_train.L_book
        self.d_book = self.dataset_train.d_book

        #self.split_train_val(self.val_split)
        self.dataset_val = LOBSTER_Dataset(
            self.val_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_cache_files=self.n_cache_files,
            randomize_offset=False,
            book_files=self.val_book_files,
            use_simple_book=self.use_simple_book,
            book_transform=self.book_transform,
            book_depth=self.book_depth,
            return_raw_msgs=self.return_raw_msgs,
        )

        self.dataset_test = LOBSTER_Dataset(
            self.test_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_cache_files=self.n_cache_files,
            randomize_offset=False,
            book_files=self.test_book_files,
            use_simple_book=self.use_simple_book,
            book_transform=self.book_transform,
            book_depth=self.book_depth,
            return_raw_msgs=self.return_raw_msgs,
        )

    def reset_train_offsets(self):
        """ reset the train dataset to a new random offset
            (e.g. every training epoch)
            keeps the same validation set and removes validation
            indices from training set
        """
        # use a new seed for the train dataset to
        # get a different random offset for each sequence for each epoch
        self.dataset_train = LOBSTER_Dataset(
            self.train_files,
            n_messages=self.n_messages,
            mask_fn=self.mask_fn,
            seed=self.rng.randint(0, sys.maxsize),
            n_cache_files=self.n_cache_files,
            randomize_offset=True,
            book_files=self.train_book_files,
            use_simple_book=self.use_simple_book,
            book_transform=self.book_transform,
            book_depth=self.book_depth,
            return_raw_msgs=self.return_raw_msgs,
        )
    #     indices = [i for i in range(len(self.dataset_train)) if i not in self.val_indices]
    #     # remove the first and last val sequence to avoid overlapping observations
    #     indices.pop(self.val_indices[0])
    #     indices.pop(self.val_indices[-1])
    #     self.dataset_train = LOBSTER_Subset(self.dataset_train, indices)
    
    # def split_train_val(self, val_split):
    #     """ takes one consecutive sequence as validation data,
    #         dropping the previous and next sequence to avoid overlapping observations
    #         with the training data.
    #     """
    #     rng = random.Random(self.seed)
    #     indices = list(range(len(self.dataset_train))) #np.arange(len(self.dataset_train))
    #     val_len = int(np.ceil(val_split * len(indices)))
        
    #     # randomly pick validation indices
    #     #val_indices = [indices.pop(rng.randrange(0, len(indices))) for _ in range(val_len)]
        
    #     val_start = rng.randrange(0, len(indices) - val_len)
    #     self.val_indices = [indices.pop(i) for i in range(val_start, val_start + val_len)]
    #     # remove previous and subsequent sequence from training data
    #     # this is to avoid overlapping observations with the validation data
    #     # when offsets are reset
    #     if val_start > 0:
    #         indices.pop(val_start - 1)
    #     if val_start + val_len < len(indices):
    #         indices.pop(val_start + val_len)

    #     self.dataset_val = LOBSTER_Subset(self.dataset_train, self.val_indices)
    #     self.dataset_train = LOBSTER_Subset(self.dataset_train, indices)

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
