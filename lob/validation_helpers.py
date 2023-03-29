from lob.encoding import Message_Tokenizer, Vocab
import jax
from jax import nn
import jax.numpy as np
from functools import partial

v = Vocab()

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
    if np.any(idx > Message_Tokenizer.MSG_LEN - 1):
        raise ValueError("Index ({}) must be less than {}".format(idx, Message_Tokenizer.MSG_LEN))
    field_i = np.searchsorted(Message_Tokenizer.TOK_DELIM, idx, side='right')
    return [Message_Tokenizer.FIELDS[i] for i in field_i]

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
    return pred[..., ::-1].argsort(axis=-1)[correct_mask]

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
    #print(p.shape)
    #print(idx.shape)
    return jax.random.choice(rng, idx, p=p)

#@partial(np.vectorize, signature="(n)->()")
#def test(a):
#    return np.argmax(a)

