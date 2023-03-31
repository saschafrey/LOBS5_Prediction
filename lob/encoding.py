import numpy as np
import pandas as pd
import jax.numpy as jnp


class Vocab:

    MASK_TOK = 0
    HIDDEN_TOK = 1
    NA_TOK = 2

    def __init__(self) -> None:
        self.counter = 3  # 0: MSK, 1: HID, 2: NAN
        self.ENCODING = {}
        self.DECODING = {}
        self.DECODING_GLOBAL = {}
        self.TOKEN_DELIM_IDX = {}

        self._add_field('time', [str(i).zfill(3) for i in range(1000)], [3,6,9,12])
        self._add_field('event_type', ['1', '2', '3', '4'], None)
        self._add_field('size', [str(i).zfill(4) for i in range(10000)], [])
        self._add_field('price', [str(i).zfill(2) for i in range(100)] + ['+', '-'], [1])
        self._add_field('direction', ['0', '1'], None)
        #self._add_field('generic', [str(i) for i in range(10)] + ['+', '-'])
        
        self._add_special_tokens()

    def __len__(self):
        return self.counter

    def _add_field(self, name, values, delim_i=None):
        enc = {val: self.counter + i for i, val in enumerate(values)}
        dec = {tok: val for val, tok in enc.items()}
        self.ENCODING[name] = enc
        self.DECODING[name] = dec
        self.DECODING_GLOBAL.update({tok: (name, val) for val, tok in enc.items()})
        self.counter += len(enc)
        self.TOKEN_DELIM_IDX[name] = delim_i

    def _add_special_tokens(self):
        for field, enc in self.ENCODING.items():
            self.ENCODING[field][Vocab.MASK_TOK] = 'MSK'
            self.ENCODING[field][Vocab.HIDDEN_TOK] = 'HID'
            self.ENCODING[field][Vocab.NA_TOK] = 'NAN'

            self.DECODING[field][Vocab.MASK_TOK] = 'MSK'
            self.DECODING[field][Vocab.HIDDEN_TOK] = 'HID'
            self.DECODING[field][Vocab.NA_TOK] = 'NAN'
        self.ENCODING['generic'] = {
            'MSK': Vocab.MASK_TOK,
            'HID': Vocab.HIDDEN_TOK,
            'NAN': Vocab.NA_TOK,
        }
        self.DECODING_GLOBAL[Vocab.MASK_TOK] = ('generic', 'MSK')
        self.DECODING_GLOBAL[Vocab.HIDDEN_TOK] = ('generic', 'HID')
        self.DECODING_GLOBAL[Vocab.NA_TOK] = ('generic', 'NAN')

class Message_Tokenizer:

    FIELDS = (
        'time',
        'event_type',
        'size',
        'price',
        'direction',
        'time_new',
        'event_type_new',
        'size_new',
        'price_new',
        'direction_new'
    )
    #FIELD_LENS = np.array((15, 1, 4, 3, 1, 15, 1, 4, 3, 1))
    TOK_LENS = np.array((5, 1, 1, 2, 1, 5, 1, 1, 2, 1))
    TOK_DELIM = np.cumsum(TOK_LENS[:-1])
    #FIELD_DELIM = np.cumsum(FIELD_LENS[:-1])
    MSG_LEN = np.sum(TOK_LENS)  #np.sum(FIELD_LENS)
    FIELD_ENC_TYPES = {
        'time': 'time', #'generic',
        'event_type': 'event_type',
        'size': 'size', #'generic',
        'price': 'price', #'generic',
        'direction': 'direction',
        'time_new': 'time', #'generic',
        'event_type_new': 'event_type',
        'size_new': 'size', #'generic',
        'price_new': 'price', #'generic',
        'direction_new': 'direction',
    }

    @staticmethod
    def get_field_from_idx(idx):
        """ Get the field of a given index (or indices) in a message
        """
        if isinstance(idx, int):
            idx = np.array([idx])
        if np.any(idx > Message_Tokenizer.MSG_LEN - 1):
            raise ValueError("Index ({}) must be less than {}".format(idx, Message_Tokenizer.MSG_LEN))
        field_i = np.searchsorted(Message_Tokenizer.TOK_DELIM, idx, side='right')
        return [Message_Tokenizer.FIELDS[i] for i in field_i]

    @staticmethod
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
        idx = tuple(jnp.array(idx).T)
        mask = jnp.zeros((Message_Tokenizer.MSG_LEN, len(v)), dtype=bool)
        mask = mask.at[idx].set(True)

        # adjustments for special tokens (no MSK, NAN, HID) allowed
        mask = mask.at[:, v.MASK_TOK].set(False)
        mask = mask.at[:, v.NA_TOK].set(False)
        mask = mask.at[:, v.HIDDEN_TOK].set(False)

        # adjustment for positions only allowing subset of field
        # e.g. +/- at start of price
        enc_type = 'price'
        allowed_toks = jnp.array([v.ENCODING[enc_type]['+'], v.ENCODING[enc_type]['-']])
        adj_col = jnp.zeros((mask.shape[1],), dtype=bool).at[allowed_toks].set(True)
        # TODO: remove hardcoding and make this more general
        mask = mask.at[(7, 17), :].set(adj_col)
        return mask
    
    #VALID_MATRIX = syntax_validation_matrix.__func__()
    
    @staticmethod
    def _generate_col_idx_by_encoder():
        """ Generates attribute dictionary col_idx_by_encoder
            with encoder type as key and a list of column (field)
            indices as value. This is used to efficiently decode tokenized
            data. 
        """
        col_idx_by_encoder = {}
        counter = 0
        for n_toks, (col, enc_type) in zip(
            Message_Tokenizer.TOK_LENS,
            Message_Tokenizer.FIELD_ENC_TYPES.items()):
            add_vals = list(range(counter, counter + n_toks))
            try:
                col_idx_by_encoder[enc_type].extend(add_vals)
            except KeyError:
                col_idx_by_encoder[enc_type] = add_vals
            counter += n_toks
        return col_idx_by_encoder

    #col_idx_by_encoder = _generate_col_idx_by_encoder.__func__()()

    def __init__(self) -> None:
        #self._generate_col_idx_by_encoder()
        pass

    def encode(self, m, vocab):
        enc = vocab.ENCODING
        #m = m.copy()

        # order ID is not used by the model
        m.drop('order_id', axis=1, inplace=True)

        for i, col in enumerate(m.columns):
            enc_type = Message_Tokenizer.FIELD_ENC_TYPES[col]
            #print(col)
            #print(enc_type)
            #print(col)
            m[col] = self._encode_col(
                m[col],
                enc=enc[enc_type],
                n_toks=Message_Tokenizer.TOK_LENS[i],
                delim_i=vocab.TOKEN_DELIM_IDX[enc_type])
        # concat all lists into single column
        m = m.sum(axis=1)
        # return as numpy array
        return np.array(m.to_list())

    def _encode_col(self, col, enc, n_toks, delim_i=None):
        def _encode_field(num):
            if pd.isnull(num):
                return [Vocab.NA_TOK] * n_toks
            elif not isinstance(num, str):
                num = str(int(num))
            if delim_i is not None:
                # split into tokenizable junks
                num = [num[i:j] for i, j in zip([0] + delim_i, delim_i + [None])]
            return [enc[d] for d in num]
        return col.apply(_encode_field)

    def decode(self, toks, vocab):
        str_arr = self.decode_to_str(toks, vocab)
        cols_str = np.split(str_arr, Message_Tokenizer.TOK_DELIM, axis=1)
        out_numeric = np.empty((toks.shape[0], len(cols_str)), dtype=float)
        # decode each column to float
        for i, inp in enumerate(cols_str):
            out_numeric[:, i] = self._parse_col(inp)

        return out_numeric
    
    def decode_to_str(self, toks, vocab, error_on_invalid=False):
        if toks.ndim == 1:
            toks = np.array(toks).reshape(-1, Message_Tokenizer.MSG_LEN)
        elif toks.ndim >= 2:
            toks = np.array(toks).reshape(toks.shape[0], -1, Message_Tokenizer.MSG_LEN)
        out = np.empty_like(toks, dtype='<U3')
        for dec_type, dec in vocab.DECODING.items():
            col_msk = np.zeros_like(toks, dtype=bool)
            col_msk[..., self.col_idx_by_encoder[dec_type]] = True
            for t, repl in dec.items():
                #print(((toks == t) * col_msk).shape)
                out[(toks == t) * col_msk] = repl

        if error_on_invalid:
            # left over empty strings imply invalid tokens
            err_i = np.argwhere(out == '')
            if len(err_i) > 0:
                err_toks = toks[tuple(err_i.T)]
                #err_toks = toks[out == '']
                err_fields = []
                for err_sample, err_col in err_i:
                    err_fields.append(np.searchsorted(Message_Tokenizer.TOK_DELIM, err_col, side='right'))
                e = ValueError(
                    f"Invalid tokens {err_toks} at indices {err_i} "
                    + f"for fields {[Message_Tokenizer.FIELDS[f] for f in err_fields]})")
                e.err_i = err_i
                raise e

        return out

    def _parse_col(self, inp):
        def try_parse_float(inp):
            try:
                return float(inp)
            except ValueError:
                return np.nan
        return np.array([try_parse_float(''.join(inp[i])) for i in range(inp.shape[0])])

    def validate(self, toks, vocab):
        """ checks if toks is syntactically AND semantically valid message
            returns triple of (is_valid, error location, error message)
        """
        valid_synt, res = self._validate_syntax(toks, vocab)
        if not valid_synt:
            return False, res, 'syntax error'
        valid_semant, err = self._validate_semantics(res)
        if not valid_semant:
            return False, _, err

    def _validate_syntax(self, toks, vocab):
        try:
            decoded = self.decode_to_str(toks, vocab, error_on_invalid=True)
            return True, decoded
        except ValueError as e:
            return False, e.err_i

    def _validate_semantics(self, decoded):
        ''' checks if decoded message string is semantically correct
            return tuple of (is_valid, error in field, error message)
        '''
        pass

    def invalid_toks_per_msg(self, toks, vocab):
        return (self.decode_to_str(toks, vocab) == '').sum(axis=-1)
    
    def invalid_toks_per_seq(self, toks, vocab):
        return self.invalid_toks_per_msg(toks, vocab).sum(axis=-1)

    def preproc(self, m, b, allowed_event_types=[1,2,3,4]):
        # TYPE
        # filter out only allowed event types ...
        m = m.loc[m.event_type.isin(allowed_event_types)].copy()
        # ... and corresponding book changes
        b = b.loc[m.index]

        # TIME
        # subtract opening time and convert to ns integer
        opening_s = 9.5 * 3600  # NASDAQ opens 9:30
        #closing_s = 16 * 3600   # and closes at 16:00
        m['time'] = (m['time'] - opening_s).multiply(1e9).round().astype(int).astype(str).str.zfill(15)
        
        # SIZE
        m.loc[m['size'] > 9999, 'size'] = 9999
        m['size'] = m['size'].astype(int).astype(str).str.zfill(4)

        # PRICE
        # (previous) best bid
        bb = b.iloc[:, 2].shift()
        # no truncation (large thresh.) --> 199 price levels
        # TODO: uncomment
        m.price = self._preproc_prices(m.price, bb, p_lower_trunc=-9900, p_upper_trunc=9900)
        m = m.dropna()
        m.price = m.price.astype(int).apply(self._numeric_str)

        # DIRECTION
        m.direction = ((m.direction + 1) / 2).astype(int)

        # add order changes as features
        m = self._add_orig_msg_features(m)

        return m

    def _preproc_prices(self, p, bb, p_lower_trunc=-1000, p_upper_trunc=1300):
        """ Takes prices series and best bid, encoding prices relative to best bid.
            Returns scaled price series
        """
        # encode prices relative to (previous) best bid
        p = p - bb
        # truncate price at deviation of 1000
        # min tick is 100, hence min 10-level diff is 900
        # <= 1000 covers ~99.54% on bid side, ~99.1% on ask size (GOOG)
        pct_changed = 100 * len(p.loc[p > p_upper_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices > {p_upper_trunc}")
        p.loc[p > p_upper_trunc] = p_upper_trunc
        pct_changed = 100 * len(p.loc[p < p_lower_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices < {p_lower_trunc}")
        p.loc[p < p_lower_trunc] = p_lower_trunc
        # scale prices to min ticks size differences
        p /= 100
        return p

    def _add_orig_msg_features(self, m):
        """ Changes representation of order cancellation (2) / deletion (3),
            representing them as the original message and new columns containing
            the new order details.
            This effectively does the lookup step in past data.
            TODO: lookup missing original message data from previous days' data 
        """

        m_changes = pd.merge(
            m.loc[m.event_type == 1],
            m.loc[(m.event_type == 2) | (m.event_type == 3)].reset_index(),
            how='right', on='order_id', suffixes=['', '_new']).set_index('index')
        #display(m_changes)

        # add new empty columns for order modifications
        m[m_changes.columns[-5:].values] = np.nan
        # replace order changes by original order and additional new fields
        #display(m)
        #display(m_changes)
        m.loc[m_changes.index] = m_changes
        return m

    def _numeric_str(self, num, pad=2):
        if num == 0:
            return '-00'
        elif num > 0:
            return '+' + str(num).zfill(pad)
        else:
            # minus sign counts as character
            return str(num).zfill(pad + 1)
