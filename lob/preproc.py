from __future__ import annotations
import argparse
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from lob.encoding import Vocab, Message_Tokenizer


def process_message_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        filter_above_lvl: Optional[int] = None
    ) -> None:

    v = Vocab()
    tok = Message_Tokenizer()

    assert len(message_files) == len(book_files)
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)

        col_names = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
        messages = pd.read_csv(
            m_f,
            names=col_names,
            usecols=col_names,
            index_col=False)

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
        print('<< encoding >>')
        m_ = tok.encode(m_, v)

        # save processed messages
        m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
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
        allowed_events=[1, 2, 3, 4]
    ) -> None:

    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        print(b_f)

        messages = pd.read_csv(
            m_f,
            names=['time', 'event_type', 'order_id', 'size', 'price', 'direction'],
            index_col=False)

        book = pd.read_csv(
            b_f,
            index_col=False,
            header=None
        )

        # remove disallowed order types
        messages = messages.loc[messages.event_type.isin(allowed_events)]
        book = book.loc[messages.index]

        if filter_above_lvl is not None:
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)

        # convert to n_price_series separate volume time series (each tick is a price level)
        book = process_book(book, price_levels=n_price_series)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        np.save(b_path, book, allow_pickle=True)

def process_book(
        b: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:

    # mid-price rounded to nearest tick (100)
    #p_ref = b[2]
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
    #bid_diff = b[2].div(100).diff().fillna(0).astype(int).values
    mid_diff = p_ref.div(100).diff().fillna(0).astype(int).values
    return np.concatenate([mid_diff[:, None], mybook], axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/nfs/home/peern/LOBS5/data/raw/',
		     			help="where to load data from")
    parser.add_argument("--save_dir", type=str, default='/nfs/home/peern/LOBS5/data/',
		     			help="where to save processed data")
    parser.add_argument("--filter_above_lvl", type=int,
                        help="filters down from levels present in the data to specified number of price levels")
    parser.add_argument("--n_tick_range", type=int, default=500,
                        help="how many ticks price series should be calculated")
    args = parser.parse_args()

    message_files = sorted(glob(args.data_dir + '*message*.csv'))
    book_files = sorted(glob(args.data_dir + '*orderbook*.csv'))

    print('found', len(message_files), 'message files')
    print('found', len(book_files), 'book files')
    print()

    print('processing messages...')
    process_message_files(message_files, book_files, args.save_dir, filter_above_lvl=args.filter_above_lvl)
    print()
    
    print('processing books...')
    process_book_files(
        message_files,
        book_files,
        args.save_dir,
        filter_above_lvl=args.filter_above_lvl,
        n_price_series=args.n_tick_range
    )
    print('DONE')
