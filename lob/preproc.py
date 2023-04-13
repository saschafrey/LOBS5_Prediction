import numpy as np
import pandas as pd


def process_book(
        b: pd.DataFrame,
        price_levels: int
    ):
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