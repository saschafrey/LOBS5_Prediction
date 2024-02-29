from __future__ import annotations
import jax
import jax.numpy as jnp
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from decimal import Decimal
from functools import partial
# import lob.encoding as encoding

import pyarrow as pa
import pyarrow.parquet as pq


import sys,os
current_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from lob.encoding import Vocab, Message_Tokenizer


@partial(jax.jit, static_argnums=(1, 2))
@partial(
    jax.vmap,
    in_axes=(0, None, None),
    out_axes=0,
)
def transform_L2_state(
        book: jax.Array, 
        price_levels: int,
        tick_size: int = 100,
    ) -> jax.Array:
    """ Transformation for data loading:
        Converts L2 book state from data to price_levels many volume
        series used as input to the model. The first element (column) of the
        input and output is the change in mid price.
        Converts sizes to negative sizes for ask side (sell orders).
    """
    delta_p_mid, book = book[:1], book[1:]
    book = book.reshape((-1,2))
    mid_price = jnp.ceil((book[0, 0] + book[1, 0]) / (2*tick_size)).__mul__(tick_size).astype(int)
    book = book.at[:, 0].set((book[:, 0] - mid_price) // tick_size)
    # change relative prices to indices
    book = book.at[:, 0].set(book[:, 0] + price_levels // 2)
    # set to out of bounds index, so that we can use -1 to indicate nan
    # out of bounds will be ignored in setting value in jax
    book = jnp.where(book < 0, -price_levels-1, book)

    mybook = jnp.zeros(price_levels, dtype=jnp.int32)
    mybook = mybook.at[book[:, 0]].set(book[:, 1])
    
    # set ask volume to negative (sell orders)
    mybook = mybook.at[price_levels // 2:].set(mybook[price_levels // 2:] * -1)
    mybook = jnp.concatenate((
        delta_p_mid.astype(np.float32),
        mybook.astype(np.float32) / 1000
    ))

    # return mybook.astype(jnp.float32) #/ divide_by
    return mybook 


def load_message_df(m_f: str) -> pd.DataFrame:
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(
        m_f,
        names=cols,
        usecols=cols,
        index_col=False,
        dtype={
            #'time': 'float64',
            'time': str,
            'event_type': 'int32',
            'order_id': 'int32',
            'size': 'int32',
            'price': 'int32',
            'direction': 'int32'
        }
    )
    messages.time = messages.time.apply(lambda x: Decimal(x))
    return messages


def process_message_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        filter_above_lvl: Optional[int] = None,
        skip_existing: bool = False,
    ) -> None:
    #Does the tokenisation of messages and saves to numpy
    v = Vocab()
    tok = Message_Tokenizer()

    assert len(message_files) == len(book_files)
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        if skip_existing and Path(m_path).exists():
            print('skipping', m_path)
            continue
        
        messages = load_message_df(m_f)

        book = pd.read_csv(
            b_f,
            index_col=False,
            header=None
        )
        assert len(messages) == len(book)

        if filter_above_lvl:
            book = book.iloc[:, :filter_above_lvl * 4]
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)
        
        print('<< pre processing >>')
        m_ = tok.preproc(messages, book)

        # save processed messages
        np.save(m_path, m_)
        print('saved to', m_path)

def get_price_range_for_level(
        book: pd.DataFrame,
        lvl: int
    ) -> pd.DataFrame:
    assert lvl > 0
    assert lvl <= (book.shape[1] // 4)
    p_range = book[[(lvl-1) * 4, (lvl-1) * 4 + 2]]
    p_range.columns = ['p_max', 'p_min']
    return p_range

def filter_by_lvl(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        lvl: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    assert messages.shape[0] == book.shape[0]
    p_range = get_price_range_for_level(book, lvl)
    messages = messages[(messages.price <= p_range.p_max) & (messages.price >= p_range.p_min)]
    book = book.loc[messages.index]
    return messages, book


def process_book_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        n_price_series: int,
        filter_above_lvl: Optional[int] = None,
        allowed_events=[1, 2, 3, 4],
        skip_existing: bool = False,
        use_raw_book_repr=False,
    ) -> None:

    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        print(b_f)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        if skip_existing and Path(b_path).exists():
            print('skipping', b_path)
            continue

        messages = load_message_df(m_f)

        book = pd.read_csv(
            b_f,
            index_col=False,
            header=None
        )

        # remove disallowed order types
        messages = messages.loc[messages.event_type.isin(allowed_events)]
        # make sure book is same length as messages
        book = book.loc[messages.index]

        if filter_above_lvl is not None:
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)

        # convert to n_price_series separate volume time series (each tick is a price level)
        if not use_raw_book_repr:
            book = process_book(book, price_levels=n_price_series)
        else:
            # prepend delta mid price column to book data
            p_ref = ((book.iloc[:, 0] + book.iloc[:, 2]) / 2).round(-2).astype(int)
            mid_diff = p_ref.div(100).diff().fillna(0).astype(int)
            book = np.concatenate((mid_diff.values.reshape(-1,1), book.values), axis=1)

        np.save(b_path, book, allow_pickle=True)

def process_book(
        b: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:

    # mid-price rounded to nearest tick (100)
    p_ref = ((b.iloc[:, 0] + b.iloc[:, 2]) / 2).round(-2).astype(int)
    b_indices = b.iloc[:, ::2].sub(p_ref, axis=0).div(100).astype(int)
    b_indices = b_indices + price_levels // 2
    b_indices.columns = list(range(b_indices.shape[1]))
    vol_book = b.iloc[:, 1::2].copy()
    # convert sell volumes (ask side) to negative
    vol_book.iloc[:, ::2] = vol_book.iloc[:, ::2].mul(-1)
    vol_book.columns = list(range(vol_book.shape[1]))

    # convert to book representation with volume at each price level relative to reference price (mid)
    # i.e. at each time we have a fixed width snapshot around the mid price
    # therefore movement of the mid price needs to be a separate feature (e.g. relative to previous price)

    mybook = np.zeros((len(b), price_levels), dtype=np.int32)

    a = b_indices.values
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            price = a[i, j]
            if price >= 0 and price < price_levels:
                mybook[i, price] = vol_book.values[i, j]

    # prepend column with best bid changes (in ticks)
    mid_diff = p_ref.div(100).diff().fillna(0).astype(int).values
    return np.concatenate([mid_diff[:, None], mybook], axis=1)

def process_fi_2010_files(test_files: list[str],
                          train_files: list[str],
                          save_dir: str,
                          skip_existing: bool = False,):
    numpy_tests=[]
    numpy_trains=[]
    train_path = (save_dir + 
                  train_files[0].rsplit('/', maxsplit=1)[-1][:train_files[0].rsplit('/', maxsplit=1)[-1].rfind("CF")+2] + 
                  '_proc.npy')
    test_path = (save_dir +
                  test_files[0].rsplit('/', maxsplit=1)[-1][:test_files[0].rsplit('/', maxsplit=1)[-1].rfind("CF")+2] +
                  '_proc.npy')
    if skip_existing and Path(test_path).exists() and Path(train_path).exists():
        print('Skipping fi-2010 files. Already exist')
        return None
    
    for te_f in test_files:
        np_test=np.loadtxt(te_f)
        numpy_tests.append(np_test)
    numpy_test=np.hstack(numpy_tests)
    for tr_f in train_files:
        np_train=np.loadtxt(tr_f)
        numpy_trains.append(np_train)
    numpy_train=np.hstack(numpy_trains)

    np.save(test_path, numpy_test, allow_pickle=True)
    np.save(train_path,numpy_train,allow_pickle=True)

    
    return numpy_test,numpy_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/data1/sascha/data/raw/',
		     			help="where to load data from")
    parser.add_argument("--save_dir", type=str, default='/data1/sascha/data/',
		     			help="where to save processed data")
    parser.add_argument("--filter_above_lvl", type=int, default=10,
                        help="filters down from levels present in the data to specified number of price levels")
    parser.add_argument("--n_tick_range", type=int, default=500,
                        help="how many ticks price series should be calculated")
    parser.add_argument("--skip_existing", action='store_true', default=False)
    parser.add_argument("--messages_only", action='store_true', default=False)
    parser.add_argument("--book_only", action='store_true', default=False)
    parser.add_argument("--representation", type=str, default="raw",choices={"raw","volume-image","order-flow","all"})
    parser.add_argument("--save-format", type=str, default="numpy",choices={"numpy","parquet"})
    parser.add_argument("--dataset", type=str, default="lobster",choices={"lobster","fi-2010"})


    args = parser.parse_args()

    #check to make sure something is being considered.
    assert not (args.messages_only and args.book_only)
    #fi-2010 dataset only has book data
    assert not (args.messages_only and (args.dataset=="fi-2010"))

    if args.dataset== "lobster":
        message_files = sorted(glob(args.data_dir + '*message*.csv'))
        book_files = sorted(glob(args.data_dir + '*orderbook*.csv'))
        print('found', len(message_files), 'message files')
        print('found', len(book_files), 'book files')
        print()
    else:
        test_files=sorted(glob(args.data_dir + 'Test*Dst_NoAuction*.txt'))
        train_files=sorted(glob(args.data_dir + 'Train*Dst_NoAuction*.txt'))
        print('found', len(test_files), 'test files')
        print('found', len(train_files), 'train files')
        print()


    if not (args.book_only or args.dataset=="fi-2010"):
        print('processing messages...')
        process_message_files(
            message_files,
            book_files,
            args.save_dir+'lobster_proc/',
            filter_above_lvl=args.filter_above_lvl,
            skip_existing=args.skip_existing,
        )
    else:
        print('Skipping message processing...')
    print()
    
    if not (args.messages_only or args.dataset=="fi-2010"):
        print('processing books...')
        process_book_files(
            message_files,
            book_files,
            args.save_dir+'lobster_proc/',
            filter_above_lvl=args.filter_above_lvl,
            n_price_series=args.n_tick_range,
            skip_existing=args.skip_existing,
            use_raw_book_repr=args.representation,
        )
    else:
        print('Skipping book processing...')
    print()

    if args.dataset=="fi-2010":
        print("Processing f1-2010 test and train files...")
        process_fi_2010_files(
            test_files=test_files,
            train_files=train_files,
            save_dir=args.save_dir+'fi2010_proc/',
            skip_existing=args.skip_existing,
        )
    else:
        print("Not processing FI-2010 files...")
    print()

    print('DONE')