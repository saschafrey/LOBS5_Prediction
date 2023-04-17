from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

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
        if filter_above_lvl:
            book = book.iloc[:, :filter_above_lvl * 4]
            messages = filter_messages_by_lvl(messages, book, filter_above_lvl)
        
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

def filter_messages_by_lvl(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        lvl: int
    ) -> pd.DataFrame:
    p_range = get_price_range_for_level(book, lvl)
    return messages[(messages.price <= p_range.p_max) & (messages.price >= p_range.p_min)]


def process_book_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        price_levels: int,
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

        # remove diallowed order types
        messages = messages.loc[messages.event_type.isin(allowed_events)]
        book = book.loc[messages.index]

        book = process_book(book, price_levels=price_levels)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        np.save(b_path, book, allow_pickle=True)

def process_book(
        b: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:
    b_indices = b.iloc[:, ::2].sub(b[2], axis=0).div(100).astype(int)
    b_indices = b_indices + price_levels // 2 - 2  # -2 to account for average spread
    b_indices.columns = list(range(b_indices.shape[1]))
    vol_book = b.iloc[:, 1::2]
    vol_book.columns = list(range(vol_book.shape[1]))

    # convert to book representation with volume at each price level relative to best bid
    # i.e. at each time we have a fixed width snapshot around the best bid
    # therefore movement of the mid price needs to be a separate feature (e.g. relative to previous price)

    mybook = np.zeros((len(b), price_levels), dtype=np.int32)

    a = b_indices.values
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            price = a[i, j]
            if price >= 0 and price < price_levels:
                mybook[i, price] = vol_book.values[i, j]

    # prepend column with best bid changes (in ticks)
    bid_diff = b[2].div(100).diff().fillna(0).astype(int).values
    return np.concatenate([bid_diff[:, None], mybook], axis=1)
