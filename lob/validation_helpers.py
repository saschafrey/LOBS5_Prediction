from typing import Optional, Tuple, Union
from lob.encoding import Message_Tokenizer, Vocab
import jax
from jax import nn
from jax.random import PRNGKeyArray
import flax
from flax.training.train_state import TrainState
import jax.numpy as np
from functools import partial
import numpy as onp

from lob.lobster_dataloader import LOBSTER_Dataset

v = Vocab()


def syntax_validation_matrix():
    """ Create a matrix of shape (MSG_LEN, VOCAB_SIZE) where a
        True value indicates that the token is valid for the location
        in the message.
    """
    v = Vocab()

    idx = []
    for i in range(Message_Tokenizer.MSG_LEN):
        field = Message_Tokenizer.get_field_from_idx(i)
        decoder_key = Message_Tokenizer.FIELD_ENC_TYPES[field[0]]
        for tok, val in v.DECODING[decoder_key].items():
            idx.append([i, tok])
    idx = tuple(np.array(idx).T)
    mask = np.zeros((Message_Tokenizer.MSG_LEN, len(v)), dtype=bool)
    mask = mask.at[idx].set(True)

    """
    # adjustment for positions only allowing subset of field
    # e.g. +/- at start of price
    enc_type = 'price'
    allowed_toks = np.array([v.ENCODING[enc_type]['+'], v.ENCODING[enc_type]['-']])
    adj_col = np.zeros((mask.shape[1],), dtype=bool).at[allowed_toks].set(True)
    i_slice = get_idx_from_field(enc_type)
    mask = mask.at[slice(i_slice), :].set(adj_col)

    enc_type = 'event_type'
    # in original event type, only new messages and executions allowed (no cancels or deletions)
    allowed_toks = np.array([v.ENCODING[enc_type]['1'], v.ENCODING[enc_type]['4']])
    adj_col = np.zeros((mask.shape[1],), dtype=bool).at[allowed_toks].set(True)
    i_slice = get_idx_from_field(enc_type)
    mask = mask.at[slice(i_slice), :].set(adj_col)
    """

    # adjustment for positions only allowing subset of field
    # e.g. +/- at start of price
    i, _ = get_idx_from_field("price")
    mask = update_allowed_tok_slice(mask, i, ['+', '-'])
    i, _ = get_idx_from_field("price_new")
    mask = update_allowed_tok_slice(mask, i, ['+', '-'])
    # only new messages and executions allowed in original message
    # and only cancels or deletions in modified message
    i, _ = get_idx_from_field("event_type")
    mask = update_allowed_tok_slice(mask, i, ['1', '4'])
    i, _ = get_idx_from_field("event_type_new")
    mask = update_allowed_tok_slice(mask, i, ['2', '3'])

    # adjustments for special tokens (no MSK or HID) allowed
    # NA always allowed
    mask = mask.at[:, v.MASK_TOK].set(False)
    mask = mask.at[:, v.HIDDEN_TOK].set(False)
    mask = mask.at[:, v.NA_TOK].set(True)

    return mask

def update_allowed_tok_slice(mask, i, allowed_toks):
    field = get_field_from_idx(i)
    enc_type = Message_Tokenizer.FIELD_ENC_TYPES[field[0]]
    allowed_toks = np.array([v.ENCODING[enc_type][t] for t in allowed_toks])
    adj_col = np.zeros((mask.shape[1],), dtype=bool).at[allowed_toks].set(True)
    mask = mask.at[i, :].set(adj_col)
    return mask

def is_tok_valid(tok, field, vocab):
    tok = tok.tolist()
    if isinstance(field, str):
        return tok in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]]
    else:
        return [t in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[f]] 
                for t, f in zip(tok, field)]

def get_masked_idx(seq):
    """ Get the indices of the masked tokens in a given input (batched or not)
    """
    if seq.ndim == 1:
        seq = seq.reshape(-1, Message_Tokenizer.MSG_LEN)
    elif seq.ndim == 2:
        seq = seq.reshape(seq.shape[0], -1, Message_Tokenizer.MSG_LEN)
    return np.argwhere(seq == v.MASK_TOK)

def get_field_from_idx(idx):
    """ Get the field of a given index (or indices) in a message
    """
    return Message_Tokenizer.get_field_from_idx(idx)

def get_idx_from_field(field):
    field_i = onp.argwhere(onp.array(Message_Tokenizer.FIELDS) == field).flatten()[0]
    return LOBSTER_Dataset._get_tok_slice_i(field_i)

def get_masked_fields(inp_maybe_batched):
    """ Get the fields of the masked tokens in a given input (batched or not)
    """
    mask_pos = get_masked_idx(inp_maybe_batched)
    return get_field_from_idx(mask_pos[..., -1])

def get_valid_toks_for_field(fields):
    """ Get the valid labels for given fields
    """
    return tuple(tuple(
        v.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]].keys())
          for field in fields)

def get_valid_toks_for_input(inp_maybe_batched):
    """ Get the valid labels for a given input (batched or not)
    """
    fields = get_masked_fields(inp_maybe_batched)
    return get_valid_toks_for_field(fields)

def valid_prediction_mass(pred, fields, top_n=None):
    """ for a predicted distribution over tokens get the total mass of the
        syntactically valid labels
        top_n: 
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    assert (len(fields) == pred.shape[0])
    valid_toks = get_valid_toks_for_field(fields)
    dim_0_i = [i for i, tok_list in enumerate(valid_toks) for tok in tok_list]
    dim_1_i = [tok for tok_list in valid_toks for tok in tok_list]
    mask_valid = np.zeros_like(pred)
    mask_valid = mask_valid.at[dim_0_i, dim_1_i].set(1)

    if top_n is not None:
        mask_top_n = mask_n_highest(pred, top_n)
        mask_valid = mask_valid * mask_top_n
        top_n_mass = np.sum(np.exp(pred) * mask_top_n, axis=1)
    else:
        top_n_mass = 1.

    return (np.sum(np.exp(pred) * mask_valid, axis=1)) / top_n_mass

def mask_n_highest(a, n):
    """ Return a mask for the n highest values in the last axis
        for a given array
    """
    n_th_largest = np.sort(a, axis=-1)[..., -n]
    # add leading dimensions to match pred
    n_th_largest = n_th_largest.reshape((-1,) + (1,)*(a.ndim-1))
    mask_top_n = np.zeros_like(a, dtype=bool)
    #mask_top_n = mask_top_n.at[a >= n_th_largest].set(True)
    mask_top_n = np.where(a >= n_th_largest, True, False)
    return mask_top_n

def pred_rank(pred, labels):
    """ Get the rank of the correct label in the predicted distribution.
        Lower is better (0 is correct prediction).
    """
    correct_mask = nn.one_hot(labels.astype(int), pred.shape[-1]).astype(bool)
    # ::-1 sorts in descending order (0 is highest rank)
    a = pred.argsort(axis=-1)
    ranks = a[..., ::-1].argsort(axis=-1)
    return ranks[correct_mask]

def fill_predicted_toks(seq, pred, top_n=1, rng=None):
    """ Set the predicted token in the given sequence
        when top_n=1, the argmax is used, otherwise a random sample
        from the top_n highest scores is used (propotional to the score)
        rng cannot be None when top_n > 1
    """
    if top_n == 1:
        vals = pred.argmax(axis=-1)
    else:
        vals = sample_pred(pred, top_n, rng)
    return seq.at[seq == v.MASK_TOK].set(vals)

#@partial(np.vectorize, signature="(n),(),(n)->()")
@partial(jax.vmap, in_axes=(0, None, 0))
def sample_pred(pred, top_n, rng):
    """ Sample from the top_n predicted labels
    """
    mask_top_n = mask_n_highest(pred, top_n)
    idx = np.arange(pred.shape[0]).reshape(pred.shape)
    p = pred * mask_top_n
    p = p / p.sum(axis=-1, keepdims=True)
    return jax.random.choice(rng, idx, p=p)

def append_hid_msg(seq):
    """ Append a new empty (HID token) message to a sequence
        removing first message to keep seq_len constant
    """
    l = Message_Tokenizer.MSG_LEN
    return np.concatenate([seq[l:], np.full((Message_Tokenizer.MSG_LEN,), Vocab.HIDDEN_TOK)])

def mask_last_msg_in_seq(
        seq: np.ndarray,
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    l = Message_Tokenizer.MSG_LEN
    assert (i >= -l) and (i < l), "i must be in [-MSG_LEN, MSG_LEN)"
    if i >= 0:
        i += len(seq) - l
    y = seq[i]
    return seq.at[i].set(v.MASK_TOK), y

@partial(jax.jit, static_argnums=(3, 4))
def predict(
        batch_inputs,
        batch_integration_timesteps,
        state,
        model,
        batchnorm,
    ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             )

    return logits

def filter_valid_pred(pred, valid_mask):
    """ Filter the predicted distribution to only include valid tokens
    """
    pred = pred * valid_mask
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred

def pred_next_tok(
        seq,
        state,
        model,
        batchnorm,
        sample_top_n,
        mask_i,
        rng,
        vocab_len,
        new_msg=False,
        valid_mask=None,  # if given, sample only from syntactically valid tokens
    ):
    """ Predict the next token with index i of the last message in the sequence
        if new_msg=True, a new empty message is appended to the sequence
        Returns the updated sequence
        TODO: add flag to only sample from syntactically valid tokens
    """

    # create masked message for prediction
    if new_msg:
        seq = append_hid_msg(seq)
    seq, _ = mask_last_msg_in_seq(seq, mask_i)
    # inference
    integration_timesteps = np.ones((1, len(seq)))
    seq_onehot = nn.one_hot(
        np.expand_dims(seq, axis=0), vocab_len).astype(float)
    logits = predict(
        seq_onehot,
        integration_timesteps, state, model, batchnorm)
    if valid_mask is not None:
        logits = filter_valid_pred(logits, valid_mask)
    # update sequence
    # note: rng arg expects one element per batch element
    seq = fill_predicted_toks(seq, logits, sample_top_n, np.array([rng]))
    return seq



def pred_msg(
        seq: np.ndarray,
        n_messages: int,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
        rng: PRNGKeyArray,
        valid_mask_array: Optional[jax.Array] = None
    ) -> np.ndarray:

    valid_mask = None
    l = Message_Tokenizer.MSG_LEN
    for m_i in range(n_messages):
        new_msg = True
        for i in range(l):
            if valid_mask_array is not None:
                valid_mask = valid_mask_array[i]
            seq = pred_next_tok(
                seq,
                state,
                model,
                batchnorm,
                sample_top_n=5,
                mask_i=i,
                new_msg=new_msg,
                vocab_len=len(v),
                rng=rng,
                valid_mask=valid_mask,
            )
            new_msg = False
    return seq

def validate_msg(
        msg: np.ndarray,
        tok: Message_Tokenizer,
        vocab: Vocab,
    ) -> bool:
    """ Validate a message's internal semantics
        Assumes the message is syntactically valid (allowed toks in all places)
        Returns True if valid
    """
    assert len(msg) == Message_Tokenizer.MSG_LEN
    err_count = 0

    msg_dec = tok.decode(msg, vocab).flatten()
    fields = {fname: i for i, fname in enumerate(Message_Tokenizer.FIELDS)}
    
    time = msg_dec[fields['time']]
    event_type = msg_dec[fields['event_type']]
    event_type_new = msg_dec[fields['event_type_new']]
    price = msg_dec[fields['price']]
    direction = msg_dec[fields['direction']]

    # if NA in second half, needs to be all NA
    nas = np.isnan(msg[len(msg)//2:])
    #nas = (msg[len(msg)//2:] == Vocab.NA_TOK)
    err = np.any(nas) and not np.all(nas)
    err_count += err
    if err:
        print("NAs must be in second half of message")

    # decode message to str repr
    #msg_str = tok.decode_to_str(msg, vocab).flatten()
    # time within opening hours
    #time = int(''.join(msg_str[slice(*get_idx_from_field("time"))]))
    err = time > 57600000000000  # 16 * 3600 * 1e9
    err_count += err
    if err:
        print("time after opening hours")

    if event_type_new in {2, 3} and not np.isnan(direction):
        direction_new = msg_dec[fields['direction_new']]
        err = direction != direction_new
        err_count += err
        if err:
            print("direction cannot be modified")
        
    # execution on bid side must be at price lvl 0
    # note: execution of BUY order is on bid side
    if event_type == 4 and direction == 1:
        err = price != 0
        err_count += err
        if err:
            print("execution on bid side must be at price lvl 0")
    
    return bool(err_count == 0)

def find_orig_msg(
        msg: jax.Array,
        seq: jax.Array,
    ) -> Optional[jax.Array]:
    """ Finds first msg location in given seq.
        NOTE: could also find earlier msg modifications, might not be the original new message
              but we know at least that the message is in the sequence
        Returns index of first token of msg in seq and None if msg is not found
    """
    occ = find_all_msg_occurances(msg, seq)
    if len(occ) > 0:
        return occ.flatten()[0]

def find_all_msg_occurances(
        msg: jax.Array,
        seq: jax.Array,
    ) -> jax.Array:
    """ Finds ALL msg locations in given seq.
        NOTE: could also find earlier msg modifications,
              the original new message might not be included
              but we know at least that the message is in the sequence.
        Returns index of first token of msg in seq and None if msg is not found
    """
    l = Message_Tokenizer.MSG_LEN
    seq = seq.reshape((-1, Message_Tokenizer.MSG_LEN))[:, :l//2]
    return np.argwhere((seq == msg).all(axis=1))
