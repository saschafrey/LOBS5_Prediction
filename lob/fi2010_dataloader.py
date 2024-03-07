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

import lob.encoding as encoding
from lob.encoding import Vocab, Message_Tokenizer
from lob.preproc import transform_L2_state
from s5.dataloaders.base import default_data_path, SequenceDataset
from s5.utils import permutations
default_data_path = Path(__file__).parent.parent.absolute()
default_data_path = default_data_path / "data"


class FI2010_Dataset(Dataset):

    def __init__(
            self,
            data,
            pred_horizon, #should be an index in [0,1,2,3,4]
            input_length,
            ) -> None:

        assert len(data) > 0

        self.data=data
        self.pred_horizon=pred_horizon
        self.input_length=input_length

        x=self._get_features(data)
        y=self._get_labels(data)

        x,y = self._prepare_ranges(x,y,self.input_length) 
        #select which horizon is of interest
        #turn labels of 1,2,3 into 0,1,2 
        y=y[:,pred_horizon]-1
        self.length=len(x)

        self.x,self.y=self._jax_data(x,y)

    def _get_features(self,data,features='book'):
        """Takes the orderbook features (4x10) and 
        transposes to have rows be events and cols be features.        
        """
        assert features in ['all','book','derived']

        if features == 'book':
            df1=data[:40,:].T
        elif features == 'all':
            df1=data[:-5,:].T
        elif features == 'derived':
            df1=data[40:-5,:].T
        return np.array(df1)
    
    def _get_labels(self,data):
        labels=data[-5:,:].T
        return labels
    
    def _prepare_ranges(self,X,Y,input_length):
        [events,features]=X.shape
        dX=np.array(X)
        label=np.array(Y)

        label=label[(input_length-1):events]
        input=np.zeros((events-input_length+1,
                        input_length,
                        features))
        for i in range(input_length,events+1):
            input[i-input_length] = dX[i-input_length:i ,
                                       :]
        return input,label
    
    def _jax_data(self,x,y):
        X=jnp.array(x)
        #X=jnp.expand_dims(X,1)
        Y=jnp.array(y)
        #Y=one_hot(Y,num_classes=3)
        return X,Y

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class FI2010(SequenceDataset):
    _name_ = "fi-2010"
    l_output = 0

    _collate_arg_names = [] #['book_data'] #['timesteps']

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
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,  # For train/val split
            "mask_fn": None,
            "use_book_data": False,
            "use_simple_book" : False,
            "book_transform": False,
            "n_cache_files": 0,
            "book_depth": 500,
            "return_raw_msgs": False,
            "horizon": 10,
            "horizon_type": 'messages'
        }

    def setup(self):
        train_files=sorted(glob(str(self.data_dir) + '/Train*Dst_NoAuction*.npy'))
        assert len(train_files) > 0, f'no train files found in {self.data_dir}'
        assert len(train_files) == 1, f'Expecting only one train file in {self.data_dir}'

        test_files=sorted(glob(str(self.data_dir) + '/Test*Dst_NoAuction*.npy'))
        assert len(test_files) == len(test_files) , f'Test file length in {self.data_dir} doesnt match with train legth'

        data_tr=np.load(train_files[0])
        data_te=np.load(test_files[0])

        train_data=data_tr[:, :int(np.floor(data_tr.shape[1] * (1-self.val_split)))]
        val_data=data_tr[:, int(np.floor(data_tr.shape[1] * (1-self.val_split))):]
        test_data=data_te


        horizon_map={
            10 : 0,
            20 : 1,
            30: 2,
            50: 3,
            100: 4
        }
        horizon_index=horizon_map[self.pred_horizon]

        self.dataset_train = FI2010_Dataset(
            data=train_data,
            pred_horizon=horizon_index,
            input_length=self.input_length,
        )

        #self.d_input = self.dataset_train.shape[-1]
        self.d_input = self.dataset_train.x.shape[-1]
        self.d_output = 1
        # sequence length
        self.L = self.input_length
        # book sequence lengths and dimension (number of levels + 1)
        self.L_book = 0
        self.d_book = 0

        self.dataset_val = FI2010_Dataset(
            data=val_data,
            pred_horizon=horizon_index,
            input_length=self.input_length,
        )
        self.dataset_test = FI2010_Dataset(
            data=test_data,
            pred_horizon=horizon_index,
            input_length=self.input_length,
        )

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"

        
    

