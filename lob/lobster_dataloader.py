""" Datasets for core experimental results """
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import pandas as pd

from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.absolute()
default_data_path = default_data_path / "data"


class LOBSTER(SequenceDataset):
    _name_ = "lobster"
    d_input = 15 # 2+7+6
    d_output = 15
    l_output = 0
    L = 500

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        '''
        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        '''

        message_files = glob.glob(self.data_dir + '/*message*')

        self.dataset_train = LOBSTER_Dataset(
            message_files,
            seq_len=self.L,
            transform=transform,
        )
        self.dataset_test = LOBSTER_Dataset(
            message_files,
            seq_len=self.L,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def split_train_val(self, val_split):
        # TODO: implement
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class LOBSTER_Dataset(Dataset):

    def __init__(self, message_files, seq_len) -> None:
        self.message_files = message_files #
        self.num_days = len(self.message_files)
        self.seq_len = seq_len
        self._seqs_per_file = np.array(
            [self._get_num_rows(f) - (self.seq_len-1) for f in message_files])
        # store at which observations files start
        self._seqs_cumsum = np.concatenate(([0], np.cumsum(self._seqs_per_file)))
        # count total number of messages once
        self._len = self._seqs_cumsum[-1]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        file_idx, seq_idx = self._get_seq_location(idx)
        #print('getting from file', file_idx, 'item', seq_idx)
        df = pd.read_csv(
            self.message_files[file_idx],
            names = ['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
            index_col = False,
            skiprows=seq_idx,
            nrows=self.seq_len
        )
        return df

    def _get_num_rows(self, file_path):
        with open(file_path) as f:
            return sum(1 for line in f)

    def _get_seq_location(self, idx):
        file_idx = np.searchsorted(self._seqs_cumsum, idx+1) - 1
        seq_idx = idx - self._seqs_cumsum[file_idx]
        return file_idx, seq_idx
