from datetime import datetime
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
from lob import train_helpers
import numpy as onp
import os
import sys
import pandas as pd
import pickle
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
from utils import debug, info

import lob.validation_helpers as valh
import lob.evaluation as eval
import lob.preproc as preproc
from lob.preproc import transform_L2_state
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderbook as job
from gym_exchange.environment.base_env.assets.action import OrderIdGenerator

ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')

@jax.jit
def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
    """"""
    orderbookLevels = len(book) // 4  # price/quantity for bid/ask
    data = jnp.array(book).reshape(int(orderbookLevels*2),2)
    newarr = jnp.zeros((int(orderbookLevels*2),8))
    initOB = newarr \
        .at[:,3].set(data[:,0]) \
        .at[:,2].set(data[:,1]) \
        .at[:,0].set(1) \
        .at[0:orderbookLevels*4:2,1].set(-1) \
        .at[1:orderbookLevels*4:2,1].set(1) \
        .at[:,4].set(0) \
        .at[:,5].set(job.INITID) \
        .at[:,6].set(34200) \
        .at[:,7].set(0).astype('int32')
    return initOB


def df_msgs_to_jnp(m_df: pd.DataFrame) -> jnp.ndarray:
    """"""
    m_df = m_df.copy()
    cols = ['Time', 'Type', 'OrderID', 'Quantity', 'Price', 'Side']
    if m_df.shape[1] == 7:
        cols += ["TradeID"]
    m_df.columns = cols
    m_df['TradeID'] = 0  #  TODO: should be TraderID for multi-agent support
    col_order=['Type','Side','Quantity','Price','TradeID','OrderID','Time']
    m_df = m_df[col_order]
    m_df = m_df[(m_df['Type'] != 6) & (m_df['Type'] != 7) & (m_df['Type'] != 5)]
    time = m_df["Time"].astype('string').str.split('.',expand=True)
    m_df[["TimeWhole","TimeDec"]] = time.astype('int32')
    m_df = m_df.drop("Time", axis=1)
    mJNP = jnp.array(m_df)
    return mJNP

@jax.jit
def msg_to_jnp(
        m_raw: jax.Array,
    ) -> jax.Array:
    """ Select only the relevant columns from the raw messages
        and rearrange for simulator.
    """
    m = m_raw.copy()
    
    return jnp.array([
        m[EVENT_TYPE_i],
        (m[DIRECTION_i] * 2) - 1,
        m[SIZE_i],
        m[PRICE_ABS_i],
        0, # TradeID
        m[ORDER_ID_i],
        m[TIMEs_i],
        m[TIMEns_i],
    ])

msgs_to_jnp = jax.jit(jax.vmap(msg_to_jnp))

# NOTE: cannot jit due to side effects --> resolve later
def reset_orderbook(
        b: OrderBook,
        l2_book: Optional[Union[pd.Series, onp.ndarray]] = None,
    ) -> None:
    """"""
    b.orderbook_array = b.orderbook_array.at[:].set(-1)
    if l2_book is not None:
        msgs = init_msgs_from_l2(l2_book)
        b.process_orders_array(msgs)

def copy_orderbook(
        b: OrderBook
    ) -> OrderBook:
    b_copy = OrderBook(price_levels=b.price_levels, orderQueueLen=b.orderQueueLen)
    b_copy.orderbook_array = b.orderbook_array.copy()
    return b_copy

def get_sim(
        init_l2_book: jax.Array,
        replay_msgs_raw: jax.Array,
        sim_book_levels: int,
        sim_queue_len: int,
    ) -> Tuple[OrderBook, jax.Array]:
    """"""
    # reset simulator
    sim = OrderBook(sim_book_levels, sim_queue_len)
    # init simulator at the start of the sequence
    reset_orderbook(sim, init_l2_book)
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    trades = sim.process_orders_array(replay)
    return sim, trades


def get_sim_msg(
        pred_msg_enc: jax.Array,
        m_seq: jax.Array,
        m_seq_raw: jax.Array,
        sim: OrderBook,
        mid_price: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[jax.Array], Optional[Dict[str, Any]]]:
    """"""
    # decoded predicted message
    # pred_msg = tok.decode(pred_msg_enc, v).squeeze()
    pred_msg = encoding.decode_msg(pred_msg_enc, encoder)
    debug('decoded predicted message:', pred_msg)

    new_part = pred_msg[: Message_Tokenizer.N_NEW_FIELDS]
    ref_part = pred_msg[Message_Tokenizer.N_NEW_FIELDS: ]

    if onp.isnan(new_part).any():
        debug('new_part contains NaNs', new_part)
        return None, None, None

    event_type = pred_msg[EVENT_TYPE_i]
    quantity = pred_msg[SIZE_i]
    side = pred_msg[DIRECTION_i]
    rel_price = pred_msg[PRICE_i]
    delta_t_s = pred_msg[DTs_i]
    delta_t_ns = pred_msg[DTns_i]
    time_s = pred_msg[TIMEs_i]
    time_ns = pred_msg[TIMEns_i]

    # NEW LIMIT ORDER
    if event_type == 1:
        sim_msg, msg_corr, raw_dict = get_sim_msg_new(
            # sim,
            mid_price,
            event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns, #delta_t, time,
            new_order_id,
            tick_size,
            encoder,
        )

    # modification / deletion / execution of existing order
    else:
        rel_price_ref = pred_msg[PRICE_REF_i]
        quantity_ref = pred_msg[SIZE_REF_i]
        time_s_ref = pred_msg[TIMEs_REF_i]
        time_ns_ref = pred_msg[TIMEns_REF_i]

        # cancel / delete
        if event_type == 2 or event_type == 3:
            sim_msg, msg_corr, raw_dict = get_sim_msg_mod(
                pred_msg_enc,
                event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns,
                rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
                mid_price,
                m_seq,
                m_seq_raw,
                sim,
                #tok,
                #v,
                tick_size,
                encoder,)

        # modify
        elif event_type == 4:
            sim_msg, msg_corr, raw_dict = get_sim_msg_exec(
                pred_msg_enc,
                event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns,
                rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
                mid_price,
                m_seq,
                m_seq_raw,
                new_order_id,
                sim,
                #tok,
                #v,
                tick_size,
                encoder,
            )

        # Invalid type of modification
        else:
            return None, None, None
                
    return sim_msg, msg_corr, raw_dict

# event_type, side, quantity, price, trade(r)_id, order_id, time_s, time_ns
@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
    ):
    """ NOTE: trade(r) ID is set to 0
    """
    return jnp.array([
        event_type,
        (side * 2) - 1,
        quantity,
        price,
        0, # trade_id
        order_id,
        time_s,
        time_ns,
    ])

@jax.jit
def construct_raw_msg(
        oid: Optional[int] = encoding.NA_VAL,
        event_type: Optional[int] = encoding.NA_VAL,
        direction: Optional[int] = encoding.NA_VAL,
        price_abs: Optional[int] = encoding.NA_VAL,
        price: Optional[int] = encoding.NA_VAL,
        size: Optional[int] = encoding.NA_VAL,
        delta_t_s: Optional[int] = encoding.NA_VAL,
        delta_t_ns: Optional[int] = encoding.NA_VAL,
        time_s: Optional[int] = encoding.NA_VAL,
        time_ns: Optional[int] = encoding.NA_VAL,
        price_ref: Optional[int] = encoding.NA_VAL,
        size_ref: Optional[int] = encoding.NA_VAL,
        time_s_ref: Optional[int] = encoding.NA_VAL,
        time_ns_ref: Optional[int] = encoding.NA_VAL,
    ):
    msg_raw = jnp.array([
        oid,
        event_type,
        direction,
        price_abs,
        price,
        size,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
        price_ref,
        size_ref,
        time_s_ref,
        time_ns_ref,
    ])
    return msg_raw

@jax.jit
def get_sim_msg_new(
        mid_price: int,
        event_type: int,
        quantity: int,
        side: int,
        rel_price: int,
        delta_t_s: int,
        delta_t_ns: int,
        time_s: int,
        time_ns: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]:
        
        # new limit order
        debug('NEW LIMIT ORDER')
        # convert relative to absolute price
        price = mid_price + rel_price * tick_size

        sim_msg = construct_sim_msg(
            1,
            side,
            quantity,
            price,
            new_order_id,
            time_s,
            time_ns,
        )
        msg_corr = construct_raw_msg(
            oid=encoding.NA_VAL,
            event_type=event_type,
            direction=side,
            price_abs=encoding.NA_VAL,
            price=rel_price,
            size=quantity,
            delta_t_s=delta_t_s,
            delta_t_ns=delta_t_ns,
            time_s=time_s,
            time_ns=time_ns,
        )

        # encode corrected message
        msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]

        nan_part = jnp.array((Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN) * [Vocab.NA_TOK])
        msg_corr = jnp.concatenate([msg_corr, nan_part])

        # create raw message to update raw data sequence
        msg_raw = encoding.decode_msg(msg_corr, encoder)
        ORDER_ID_i = 0
        PRICE_ABS_i = 3
        msg_raw = msg_raw.at[ORDER_ID_i].set(new_order_id)
        msg_raw = msg_raw.at[PRICE_ABS_i].set(price)
        debug(encoding.repr_raw_msg(msg_raw))

        return sim_msg, msg_corr, msg_raw

@jax.jit
def rel_to_abs_price(
        p_rel: jax.Array,
        best_bid: jax.Array,
        best_ask: jax.Array,
        tick_size: int = 100,
    ) -> jax.Array:

    p_ref = (best_bid + best_ask) / 2
    p_ref = ((p_ref // tick_size) * tick_size).astype(jnp.int32)
    return p_ref + p_rel * tick_size

@jax.jit
def construct_orig_msg_enc(
        pred_msg_enc: jax.Array,
        #v: Vocab,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
    """ Reconstructs encoded original message WITHOUT Delta t
        from encoded message string --> delta_t field is filled with NA_TOK
    """
    return jnp.concatenate([
        encoding.encode(jnp.array([1]), *encoder['event_type']),
        pred_msg_enc[slice(*valh.get_idx_from_field('direction'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('price_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size_ref'))],
        # NOTE: no delta_t here
        jnp.full(
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_s']] + \
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_ns']],
            Vocab.NA_TOK
        ),
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns_ref'))],
    ])

@jax.jit
def convert_msg_to_ref(
        pred_msg_enc: jax.Array,
    ) -> jax.Array:
    """ Converts encoded message to reference message part,
        i.e. (price, size, time) tokens
    """
    return jnp.concatenate([
        pred_msg_enc[slice(*valh.get_idx_from_field('price'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns'))],
    ])

# TODO: resolve control flow to be able to jit function
#@jax.jit
def get_sim_msg_mod(
        pred_msg_enc: jax.Array,
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        delta_t_s: int,
        delta_t_ns: int,
        time_s: int,
        time_ns: int,

        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,

        mid_price: int,
        m_seq: jax.Array,
        m_seq_raw: jax.Array,
        sim: OrderBook,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]:

    debug('ORDER CANCEL / DELETE')
    REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

    # the actual price of the order to be modified
    p_mod_raw = mid_price + rel_price * tick_size

    debug('rel price', rel_price)
    debug('side', side)
    debug('removed_quantity (raw)', removed_quantity)
    debug('total liquidity at price', sim.get_volume_at_price(side, p_mod_raw))
    debug('event_type:', event_type)

    # orig order not referenced (no ref given or part missing)
    if encoding.is_special_val(rel_price_ref) \
            or encoding.is_special_val(quantity_ref) \
            or encoding.is_special_val(time_s_ref) \
            or encoding.is_special_val(time_ns_ref):
        debug('NaN ref value found')
        debug('rel_price_ref', rel_price_ref, 'quantity_ref', quantity_ref, 'time_s_ref', time_s_ref, 'time_ns_ref', time_ns_ref)
        debug('remaining init vol at price', job.get_init_volume_at_price(sim.orderbook_array, side, p_mod_raw), p_mod_raw)
        # if no init volume remains at price, discard current message
        if job.get_init_volume_at_price(sim.orderbook_array, side, p_mod_raw) == 0:
            return None, None, None
        order_id = job.INITID
        #orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])
    
    # search for original order to get correct ID
    else:
        if job.get_volume_at_price(sim.orderbook_array, side, p_mod_raw) == 0:
            debug('No volume at given price, discarding...')
            return None, None, None

        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        # ref part is only needed to match to an order ID
        # find original msg index location in the sequence (if it exists)
        orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)
        debug('reconstruct. orig_enc \n', orig_enc)

        sim_ids = job.get_order_ids(
            sim.orderbook_array,
            sim.price_levels,
            sim.orderQueueLen)
        debug('sim IDs', sim_ids[sim_ids > 1])
        mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim.get_order_ids())
        orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, seq_mask=mask)
        
        # didn't find matching original message
        if orig_i is None:
            if job.get_init_volume_at_price(sim.orderbook_array, side, p_mod_raw) == 0:
                debug('No init volume found', side, p_mod_raw)
                debug(
                    sim.orderbook_array[side][
                        sim.orderbook_array[side,:,:,1] == p_mod_raw
                    ]
                )
                return None, None, None
            order_id = job.INITID
            # keep generated ref part, which we cannot validate
            orig_msg_found = orig_enc[-REF_LEN:]
        
        # found matching original message
        else:
            # get order ID from raw data for simulator
            ORDER_ID_i = 0
            order_id = m_seq_raw[orig_i, ORDER_ID_i]
            # found original message: convert to ref part
            EVENT_TYPE_i = 1
            if m_seq_raw[orig_i, EVENT_TYPE_i] == 1:
                orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
            # found reference to original message
            else:
                # take ref fields from matching message
                orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])

    # get remaining quantity in book for given order ID
    debug('looking for order', order_id, 'at price', p_mod_raw)
    remaining_quantity = job.get_order_by_id_and_price(
        sim.orderbook_array,
        order_id,
        p_mod_raw
    )[0]
    debug('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        remaining_quantity = job.get_init_volume_at_price(
            sim.orderbook_array,
            side,
            p_mod_raw
        )
        debug('remaining init qu.', remaining_quantity)
        # if no init volume remains at price, discard current message
        if remaining_quantity == 0:
            return None, None, None
        order_id = job.INITID
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity
        # change partial cancel to full delete
        if event_type == 2:
            event_type = 3
    # change full delete to partial cancel
    elif event_type == 3:
        event_type = 2

    debug(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')

    sim_msg = construct_sim_msg(
        event_type,
        side,
        removed_quantity,
        p_mod_raw,
        order_id,
        time_s,
        time_ns,
    )

    msg_corr = construct_raw_msg(
        oid=encoding.NA_VAL,
        event_type=event_type,
        direction=side,
        price_abs=encoding.NA_VAL,
        price=rel_price,
        size=removed_quantity,
        delta_t_s=delta_t_s,
        delta_t_ns=delta_t_ns,
        time_s=time_s,
        time_ns=time_ns,
    )

    # encode corrected message
    msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]
    msg_corr = onp.concatenate([msg_corr, orig_msg_found])

    # create raw message to update raw data sequence
    msg_raw = encoding.decode_msg(msg_corr, encoder)
    ORDER_ID_i = 0
    PRICE_ABS_i = 3
    msg_raw = msg_raw.at[ORDER_ID_i].set(order_id)
    msg_raw = msg_raw.at[PRICE_ABS_i].set(p_mod_raw)

    return sim_msg, msg_corr, msg_raw


def get_sim_msg_exec(
        pred_msg_enc: jnp.ndarray,
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        delta_t_s: int,
        delta_t_ns: int,
        time_s: int,
        time_ns: int,

        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,
        
        mid_price: int,
        m_seq: onp.ndarray,
        m_seq_raw: jax.Array,
        new_order_id: int,
        sim: OrderBook,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]:

    debug('ORDER EXECUTION')
    REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

    # the actual price of the order to be modified
    p_mod_raw = mid_price + rel_price * tick_size

    debug('event_type:', event_type)
    debug('side:', side)
    debug('removed_quantity:', removed_quantity)

    # get order against which execution is happening
    passive_order = sim.orderbook_array[side, 0, 0]
    if side == 0:
        debug('   execution on ask side (buyer initiated)')
        debug('   best ask:', passive_order[1])
    else:
        debug('   execution on bid side (seller initiated)')
        debug('   best bid:', passive_order[1])
    if p_mod_raw != passive_order[1]:
        debug('EXECUTION AT WRONG PRICE', 'gen:', p_mod_raw, 'p_passive', passive_order[1], 'correcting...')
        p_mod_raw = passive_order[1]

    remaining_quantity = passive_order[0]
    debug('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        debug('NOTHING TO EXECUTE AGAINST (empty side of book)')
        return None, None, None

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity

    debug(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')
    
    sim_msg = construct_sim_msg(
        1,
        1 - side,
        removed_quantity,
        p_mod_raw,
        new_order_id,
        time_s,
        time_ns,
    )
    msg_corr = construct_raw_msg(
        oid=encoding.NA_VAL,
        event_type=event_type,
        direction=side,
        price_abs=encoding.NA_VAL,
        price=rel_price,
        size=removed_quantity,
        delta_t_s=delta_t_s,
        delta_t_ns=delta_t_ns,
        time_s=time_s,
        time_ns=time_ns,
    )

    # correct the order which is executed in the sequence
    order_id = passive_order[3]
    ORDER_ID_i = 0
    orig_i = onp.argwhere(m_seq_raw[:, ORDER_ID_i] == order_id)
    # found correct order
    if len(orig_i) > 0:
        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        orig_i = orig_i.flatten()[0]

        # found original message: convert to ref part
        EVENT_TYPE_i = 1
        if m_seq_raw[orig_i, EVENT_TYPE_i] == 1:
            orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
        # found reference to original message
        else:
            # take ref fields from matching message
            orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])

    # didn't find correct order (e.g. INITID)
    else:
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])

    # encode corrected message
    msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]
    msg_corr = onp.concatenate([msg_corr, orig_msg_found])

    # create raw message to update raw data sequence
    msg_raw = encoding.decode_msg(msg_corr, encoder)
    ORDER_ID_i = 0
    PRICE_ABS_i = 3
    msg_raw = msg_raw.at[ORDER_ID_i].set(order_id)
    msg_raw = msg_raw.at[PRICE_ABS_i].set(p_mod_raw)

    return sim_msg, msg_corr, msg_raw

@jax.jit
def get_invalid_ref_mask(
        m_seq_raw: jax.Array,
        p_mod_raw: int,
        sim_ids: jax.Array
    ):
    """
    """
    PRICE_ABS_i = 3
    # filter sequence to prices matching the correct price level
    wrong_price_mask = (m_seq_raw[:, PRICE_ABS_i] != p_mod_raw)
    # filter to orders still in the book: order IDs from sim
    ORDER_ID_i = 0
    not_in_book_mask = jnp.isin(m_seq_raw[:, ORDER_ID_i], sim_ids, invert=True)
    mask = not_in_book_mask | wrong_price_mask
    return mask

@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """ Adds two timestamps given as seconds and nanoseconds each (both fit in int32)
        and returns new timestamp, split into time_s and time_ns
    """
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    a_s = a_s + b_s + extra_s
    return a_s, a_ns

def generate(
        m_seq: jax.Array,
        b_seq: jax.Array,
        m_seq_raw: jax.Array,
        n_msg_todo: int,
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        rng: jax.random.PRNGKeyArray,
        sample_top_n: int = 50,
        tick_size: int = 100,
        # if eval_msgs given, also returns loss of predictions
        # e.g. to calculate perplexity
        m_seq_eval: Optional[jax.Array] = None,  
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, jax.Array]:

    id_gen = OrderIdGenerator()
    l = Message_Tokenizer.MSG_LEN
    last_start_i = m_seq.shape[0] - l
    v = Vocab()
    vocab_len = len(v)
    valid_mask_array = valh.syntax_validation_matrix()
    l2_book_states = []
    m_seq = m_seq.copy()
    b_seq = b_seq.copy()
    m_seq_raw = m_seq_raw.copy()
    num_errors = 0

    if m_seq_eval is not None:
        m_seq_eval = m_seq_eval.reshape((-1, Message_Tokenizer.MSG_LEN))
        losses = onp.zeros(m_seq_eval.shape)

    time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
    time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
    delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
    delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    # get current mid price from simulator
    ask = sim.get_best_ask()
    bid = sim.get_best_bid()
    if ask > 0 and bid > 0:
        p_mid_old = (ask + bid) / 2
    elif ask < 0 and bid < 0:
        raise ValueError("No valid ask or bid price in order book")
    elif ask < 0:
        p_mid_old = bid + tick_size
    elif bid < 0:
        p_mid_old = ask - tick_size
    # round down to next valid tick
    p_mid_old = (p_mid_old // tick_size) * tick_size

    while n_msg_todo > 0:
    # for i_message in range(n_msg_todo):
        rng, rng_ = jax.random.split(rng)

        # TODO: combine into one call (returning 2 element array)
        time_init_s, time_init_ns = encoding.decode_time(
            m_seq[last_start_i + time_s_start_i: last_start_i + time_ns_end_i],
            encoder
        )

        # roll sequence one step forward
        m_seq = valh.append_hid_msg(m_seq)

        # TODO: calculating time in case where generation is not sequentially left to right
        #       --> check if delta_t complete --> calc time once

        # get next message: generate l tokens
        for mask_i in range(l):
            # calculate time once from previous time and delta_t
            if mask_i == TIME_START_I:
                # TODO: simplify --> separate function
                delta_t_s_toks = m_seq[last_start_i + delta_t_s_start_i: last_start_i + delta_t_s_end_i]
                delta_t_ns_toks = m_seq[last_start_i + delta_t_ns_start_i: last_start_i + delta_t_ns_end_i]
                debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
                delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
                delta_t_s = encoding.combine_field(delta_t_s, 3)
                delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
                delta_t_ns = encoding.combine_field(delta_t_ns, 3)

                debug('delta_t', delta_t_s, delta_t_ns)
                time_s, time_ns = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
                debug('time', time_s, time_ns)
                
                # encode time and add to sequence
                time_s = encoding.split_field(time_s, 2, 3)
                time_s_toks = encoding.encode(time_s, *encoder['time'])
                time_ns = encoding.split_field(time_ns, 3, 3)
                time_ns_toks = encoding.encode(time_ns, *encoder['time'])

                debug('time_toks', time_s_toks, time_ns_toks)
                m_seq = m_seq.at[last_start_i + time_s_start_i: last_start_i + time_ns_end_i].set(
                    jnp.hstack([time_s_toks, time_ns_toks]))
            # skip generation of time tokens
            if (mask_i >= TIME_START_I) and (mask_i < TIME_END_I):
                continue

            # syntactically valid tokens for current message position
            valid_mask = valh.get_valid_mask(valid_mask_array, mask_i)
            m_seq, _ = valh.mask_last_msg_in_seq(m_seq, mask_i)
            input = (
                one_hot(
                    jnp.expand_dims(m_seq, axis=0), vocab_len
                ).astype(float),
                jnp.expand_dims(b_seq, axis=0)
            )
            integration_timesteps = (
                jnp.ones((1, len(m_seq))), 
                jnp.ones((1, len(b_seq)))
            )
            logits = valh.predict(
                input,
                integration_timesteps, train_state, model, batchnorm)
            
            # filter out (syntactically) invalid tokens for current position
            if valid_mask is not None:
                logits = valh.filter_valid_pred(logits, valid_mask)

            # update sequence
            # note: rng arg expects one element per batch element
            rng, rng_ = jax.random.split(rng)
            m_seq = valh.fill_predicted_toks(m_seq, logits, sample_top_n, jnp.array([rng_]))

        debug(m_seq[-l:])
        # TODO: remove
        # debug('decoded:')
        # debug(
        #     encoding.repr_raw_msg(
        #         encoding.decode_msg(m_seq[-l:], encoder)
        #     )
        # )
        ### process generated message

        order_id = id_gen.step()

        debug(sim.get_L2_state())

        # update mid price if a new one exists (both some buy and sell order in book)
        ask = sim.get_best_ask()
        bid = sim.get_best_bid()
        if ask > 0 and bid > 0:
            p_mid_old = (ask + bid) / 2
            p_mid_old = (p_mid_old // tick_size) * tick_size

        # parse generated message for simulator, also getting corrected raw message
        # (needs to be encoded and overwrite originally generated message)
        sim_msg, msg_corr, msg_raw = get_sim_msg(
            m_seq[-l:],  # the generated message
            m_seq[:-l],  # sequence without generated message
            m_seq_raw[1:],   # raw data (same length as sequence without generated message)
            sim,
            mid_price=p_mid_old.astype(jnp.int32),
            new_order_id=order_id,
            tick_size=tick_size,
            encoder=encoder,
        )

        if sim_msg is None:
            info('invalid message - discarding...\n')
            num_errors += 1

            # cut away generated message and pad begginning of sequence
            # TODO: ideally the initial first message should be added to sequence again
            m_seq = onp.concatenate([
                onp.full((l,), Vocab.NA_TOK),
                m_seq[: -l]])
            continue

        info(sim_msg)

        # replace faulty message in sequence with corrected message
        m_seq = m_seq.at[-l:].set(msg_corr)
        # append new message to raw data
        m_seq_raw = jnp.concatenate([
            m_seq_raw[1:],
            jnp.expand_dims(msg_raw, axis=0)
        ])
        debug('new raw msg', encoding.repr_raw_msg(m_seq_raw[-1]))

        # feed message to simulator, updating book state
        _trades = sim.process_order_jnp(sim_msg)
        debug('trades', _trades)
        p_mid_new = (sim.get_best_ask() + sim.get_best_bid()) / 2
        p_mid_new = (p_mid_new // tick_size) * tick_size
        p_change = ((p_mid_new - p_mid_old) // tick_size).astype(jnp.int32)

        # get new book state
        book = sim.get_L2_state()
        l2_book_states.append(book)

        new_book_raw = jnp.concatenate([jnp.array([p_change]), book]).reshape(1,-1)
        new_book = preproc.transform_L2_state(new_book_raw, 500, 100)
        # update book sequence
        b_seq = jnp.concatenate([b_seq[1:], new_book])

        debug('p_change', p_change, '\n------------------------------------\\n')

        n_msg_todo -= 1

    if m_seq_eval is None:
        losses = None    
    return m_seq, b_seq, m_seq_raw, jnp.array(l2_book_states), num_errors, losses

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def calc_sequence_losses(
        m_seq,
        b_seq,
        state,
        model,
        batchnorm,
        n_inp_msgs,  # length of input sequence in messages
        vocab_len,
        valid_mask_array
    ):
    """ Takes a sequence of messages, and calculates cross-entropy loss for each message,
        based on the next message in the sequence.
    """
    @partial(jax.jit, static_argnums=(1,2))
    def moving_window(a: jax.Array, size: int, stride: int = 1):
        starts = jnp.arange(0, len(a) - size + 1, stride)
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(
                a,
                (start, *jnp.zeros(a.ndim-1, dtype=jnp.int32)),
                (size, *a.shape[1:])
            )
        )(starts)
    
    l = Message_Tokenizer.MSG_LEN

    @jax.jit
    def prep_single_inp(
            mask_i,
            na_mask,
            m_seq,
            b_seq,
        ):
        m_seq = m_seq.copy().reshape((-1, l))
        last_msg = jnp.where(
            na_mask,
            Vocab.HIDDEN_TOK,#Vocab.NA_TOK,
            m_seq[-1]
        )
        m_seq = m_seq.at[-1, :].set(last_msg).reshape(-1)
        m_seq, y = valh.mask_last_msg_in_seq(m_seq, mask_i)

        input = (
            one_hot(
                m_seq,
                vocab_len
            ).astype(jnp.float32),
            b_seq
        )
        integration_timesteps = (
            jnp.ones(len(m_seq), dtype=jnp.float32), 
            jnp.ones(len(b_seq), dtype=jnp.float32)
        )
        return input, integration_timesteps, y.astype(jnp.float32)
    prep_multi_input = jax.vmap(prep_single_inp, in_axes=(0, 0, None, None))

    @jax.jit
    def single_msg_losses(carry, inp):
        @partial(jax.jit, static_argnums=(0,))
        def na_mask_slice(last_non_masked_i):
            a = jnp.ones((l,), dtype=jnp.bool_)
            a = a.at[: last_non_masked_i+1].set(False)
            return a

        m_seq, b_seq, valid_mask = inp
        mask_idxs = jnp.concatenate([jnp.arange(0, TIME_START_I), jnp.arange(TIME_END_I, l)])
        na_masks = jnp.array([na_mask_slice(i) for i in range(TIME_START_I)] \
            + [na_mask_slice(i) for i in range(TIME_END_I, l)])

        bsz = 10
        assert 2*bsz >= mask_idxs.shape[0], f'bsz:{bsz}; msg len:{mask_idxs.shape[0]}'
        # split inference into two batches to avoid OOM
        input, integration_timesteps, y1 = prep_multi_input(mask_idxs[:bsz], na_masks[:bsz], m_seq, b_seq)
        logits1 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        input, integration_timesteps, y2 = prep_multi_input(mask_idxs[-bsz:], na_masks[-bsz:], m_seq, b_seq)
        logits2 = valh.predict(
            input,
            integration_timesteps, state, model, batchnorm)
        
        logits = jnp.concatenate([logits1, logits2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        y = jnp.concatenate([y1, y2[2*bsz - mask_idxs.shape[0] : ]], axis=0)
        
        # filter out (syntactically) invalid tokens for current position
        if valid_mask is not None:
            logits = valh.filter_valid_pred(logits, valid_mask)

        losses = train_helpers.cross_entropy_loss(logits, y)
        return carry, losses

    m_seq = m_seq.reshape((-1, l))
    inputs = (
        moving_window(m_seq, n_inp_msgs),
        moving_window(b_seq, n_inp_msgs),
        jnp.repeat(
            jnp.expand_dims(
                jnp.delete(valid_mask_array, slice(TIME_START_I, TIME_END_I), axis=0),
                axis=0
            ),
            m_seq.shape[0] - n_inp_msgs + 1,
            axis=0
        )
    )
    last_i, losses = jax.lax.scan(
        single_msg_losses,
        init=0,
        xs=inputs
    )
    return losses

def generate_single_rollout(
        m_seq_inp,
        b_seq_inp,
        m_seq_raw_inp,
        n_gen_msgs,
        sim,
        state,
        model,
        batchnorm,
        encoder,
        rng,
        m_seq_eval = None
    ):
    
    rng, rng_ = jax.random.split(rng)        
    # copy initial order book state for generation

    # generate predictions
    m_seq_gen, b_seq_gen, m_seq_raw_gen, l2_book_states, err, losses = generate(
        m_seq_inp,
        b_seq_inp,
        m_seq_raw_inp,
        n_gen_msgs,
        sim,
        state,
        model,
        batchnorm,
        encoder,
        rng_,
        sample_top_n=-1,  # sample from entire distribution
    )
    # only keep actually newly generated messages
    m_seq_raw_gen = m_seq_raw_gen[-n_gen_msgs:]

    return (
        m_seq_gen,
        b_seq_gen,
        m_seq_raw_gen, 
        {
            'event_types_gen': eval.event_type_count(m_seq_raw_gen[:, 1]),
            'num_errors': err,
            'l2_book_states': l2_book_states,
        }
    )

@partial(jax.jit, static_argnums=(5,))
def calculate_rollout_metrics(
        m_seq_raw_gen: jax.Array,
        m_seq_raw_eval: jax.Array,
        l2_book_states: jax.Array,
        l2_book_states_eval: jax.Array,
        l2_book_state_init: jax.Array,
        data_levels: int,
    ) -> Dict[str, jax.Array]:

    # arrival times
    delta_t_gen = m_seq_raw_gen[:, DTs_i].astype(jnp.float32) \
                + m_seq_raw_gen[:, DTns_i].astype(jnp.float32) / 1e9
    delta_t_gen = jnp.where(
        delta_t_gen > 0,
        delta_t_gen,
        1e-9
    )
    delta_t_eval = m_seq_raw_eval[:, DTs_i].astype(jnp.float32) \
                 + m_seq_raw_eval[:, DTns_i].astype(jnp.float32) / 1e9
    delta_t_eval = jnp.where(
        delta_t_eval > 0,
        delta_t_eval,
        1e-9
    )

    ## MID PRICE EVAL:
    # mid price at start of generation
    mid_t0 = l2_book_state_init[([0, 2],)].mean()

    # mean mid-price over J iterations
    mid_gen = jnp.mean(
        (l2_book_states[:, :, 0] + l2_book_states[:, :, 2]) / 2.,
        axis=0
    )
    # filter our mid-prices where one side is empty
    mid_gen = jnp.where(
        ((l2_book_states[:, :, 0] < 0) | (l2_book_states[:, :, 2] < 0)).any(axis=0),
        jnp.nan,
        mid_gen
    )
    rets_gen = mid_gen / mid_t0 - 1
    
    mid_eval = (l2_book_states_eval[:, 0] + l2_book_states_eval[:, 2]) / 2.
    # filter our mid-prices where one side is empty
    mid_eval = jnp.where(
        (l2_book_states_eval[:, 0] < 0) | (l2_book_states_eval[:, 2] < 0),
        jnp.nan,
        mid_eval
    )
    rets_eval = mid_eval / mid_t0 - 1
    
    # shape: (n_eval_messages, )
    mid_ret_errs = eval.mid_price_ret_squ_err(
        mid_gen, mid_eval, mid_t0)
    # compare to squared error from const prediction
    mid_ret_errs_const = jnp.square(rets_eval)
    
    ## BOOK EVAL:
    # get loss sequence using J generations and 1 evaluation
    book_losses_l1 = eval.book_loss_l1_batch(l2_book_states, l2_book_states_eval, data_levels)
    book_losses_wass = eval.book_loss_wass_batch(l2_book_states, l2_book_states_eval, data_levels)
    # compare to loss between fixed book (at t0) and actual book
    # --> as if we were predicting with the most recent observation
    book_losses_l1_const = eval.book_loss_l1(
        jnp.tile(l2_book_state_init, (l2_book_states_eval.shape[0], 1)),
        l2_book_states_eval,
        data_levels
    )
    book_losses_wass_const = eval.book_loss_wass(
        jnp.tile(l2_book_state_init, (l2_book_states_eval.shape[0], 1)),
        l2_book_states_eval,
        data_levels
    )

    metrics = {
        'delta_t_gen': delta_t_gen,
        'delta_t_eval': delta_t_eval,
        'rets_gen': rets_gen,
        'rets_eval': rets_eval,
        'mid_ret_errs': mid_ret_errs,
        'mid_ret_errs_const': mid_ret_errs_const,
        'book_losses_l1': book_losses_l1,
        'book_losses_l1_const': book_losses_l1_const,
        'book_losses_wass': book_losses_wass,
        'book_losses_wass_const': book_losses_wass_const,
    }
    return metrics

def generate_repeated_rollouts(
        num_repeats: int,
        m_seq: jax.Array,
        b_seq_pv: jax.Array,
        msg_seq_raw: jax.Array,
        book_l2_init: jax.Array,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        rng: jax.random.PRNGKeyArray,
        n_vol_series: int,
        sim_book_levels: int,
        sim_queue_len: int,
        data_levels: int,
    ):

    l2_book_states = jnp.zeros((num_repeats, n_gen_msgs, sim_book_levels * 4))
    # how many messages had to be discarded
    num_errors = jnp.zeros(num_repeats, dtype=jnp.int32)
    event_types_gen = jnp.zeros((num_repeats, 4))
    event_types_eval = jnp.zeros((num_repeats, 4))
    raw_msgs_gen = jnp.zeros((num_repeats, n_gen_msgs, 14))

    # transform book to volume image representation for model
    b_seq = jnp.array(transform_L2_state(b_seq_pv, n_vol_series, 100))

    # encoded data
    m_seq_inp = m_seq[: seq_len]
    m_seq_eval = m_seq[seq_len: ]
    b_seq_inp = b_seq[: n_msgs]
    b_seq_eval = b_seq[n_msgs: ]
    # true L2 data
    b_seq_pv_eval = jnp.array(b_seq_pv[n_msgs: ])

    # raw LOBSTER data
    m_seq_raw_inp = msg_seq_raw[: n_msgs]
    m_seq_raw_eval = msg_seq_raw[n_msgs: ]

    # initialise simulator
    sim_init, _trades = get_sim(
        book_l2_init,  # book state before any messages
        m_seq_raw_inp, # messages to replay to init sim
        sim_book_levels,  
        sim_queue_len,
    )
    # book state after initialisation (replayed messages)
    l2_book_state_init = sim_init.get_L2_state()

    # run actual messages on sim_eval (once) to compare
    sim_eval = copy_orderbook(sim_init)
    # convert m_seq_raw_eval to sim_msgs
    msgs_eval = msgs_to_jnp(m_seq_raw_eval[: n_gen_msgs])
    l2_book_states_eval, _ = sim_eval.process_orders_array_l2(msgs_eval)

    # TODO: repeat for multiple scenarios from same input to average over
    #       --> parallelise? loaded data is the same, just different rngs
    for i in range(num_repeats):
        print('ITERATION', i)
        m_seq_gen, b_seq_gen, m_seq_raw_gen, rollout_metrics = generate_single_rollout(
            m_seq_inp,
            b_seq_inp,
            m_seq_raw_inp,
            n_gen_msgs,
            copy_orderbook(sim_init),
            train_state,
            model,
            batchnorm,
            encoder,
            rng,
            m_seq_eval
        )
        event_types_gen = event_types_gen.at[i].set(rollout_metrics['event_types_gen'])
        event_types_eval = event_types_eval.at[i].set(eval.event_type_count(m_seq_raw_eval[:, 1]))
        num_errors = num_errors.at[i].set(rollout_metrics['num_errors'])
        l2_book_states = l2_book_states.at[i, :, :].set(rollout_metrics['l2_book_states'])
        raw_msgs_gen = raw_msgs_gen.at[i, :, :].set(m_seq_raw_gen)

    metrics = calculate_rollout_metrics(
        m_seq_raw_gen,
        m_seq_raw_eval,
        l2_book_states,
        l2_book_states_eval,
        l2_book_state_init,
        data_levels
    )
    metrics['l2_book_states'] = l2_book_states
    metrics['l2_book_states_eval'] = l2_book_states_eval
    metrics['num_errors'] = num_errors
    metrics['event_types_gen'] = event_types_gen
    metrics['event_types_eval'] = event_types_eval
    metrics['raw_msgs_gen'] = raw_msgs_gen
    metrics['raw_msgs_eval'] = m_seq_raw_eval

    return metrics

def sample_messages(
        n_samples: int,  # draw n random samples from dataset for evaluation
        num_repeats: int,  # how often to repeat generation for each data sample
        ds: LOBSTER_Dataset,
        rng: jax.random.PRNGKeyArray,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        n_vol_series: int = 500,
        sim_book_levels: int = 20,
        sim_queue_len: int = 100,
        data_levels: int = 10,
        save_folder: str = './tmp/'
    ):

    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples,),
        replace=False)

    all_metrics = []

    # create folder if it doesn't exist yet
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # iterate over random samples
    for i in tqdm(sample_i):

        # check if file already exists
        if os.path.isfile(save_folder + f'tmp_inference_results_dict_{i}.pkl'):
            print(f'Skipping existing sample {i}...')
            continue
        
        print(f'Processing sample {i}...')

        # 0: encoded message sequence
        # 1: prediction targets (dummy 0 here)
        # 2: book sequence (in Price, Volume format)
        # 3: raw message sequence (pandas df from LOBSTER)
        # 4: initial level 2 book state (before start of sequence)
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[int(i)]
        sequence_metrics = generate_repeated_rollouts(
            num_repeats,
            m_seq,
            b_seq_pv,
            msg_seq_raw,
            book_l2_init,
            seq_len,
            n_msgs,
            n_gen_msgs,
            train_state,
            model,
            batchnorm,
            encoder,
            rng,
            n_vol_series,
            sim_book_levels,
            sim_queue_len,
            data_levels
        )
        # save results dict as pickle file
        with open(save_folder + f'/tmp_inference_results_dict_{i}.pkl', 'wb') as f:
            pickle.dump(sequence_metrics, f)
        all_metrics.append(sequence_metrics)
    # combine metrics into single dict
    all_metrics = {
        metric: jnp.array([d[metric] for d in all_metrics])
        for metric in all_metrics[0].keys()
    }
    return all_metrics
