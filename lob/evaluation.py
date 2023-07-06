from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as onp
from functools import partial
import matplotlib.pyplot as plt

@jax.jit 
def wasserstein(p, q, order):
    # TODO: implement general case with ECDFs where sample sizes are different
    #       and also multidimensional case
    assert p.shape == q.shape
    p = jnp.sort(p)
    q = jnp.sort(q)
    return (jnp.sum(jnp.abs(p - q)**order) / p.shape[0])**(1 / order)

wasserstein_vmap = jax.jit(jax.vmap(wasserstein, in_axes=(0, 0, None)))

@jax.jit
def calc_liquidity(
        m_raw: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
    EVENT_TYPE_i = 1
    SIZE_i = 5
    liq_prov = jnp.where(
        m_raw[..., EVENT_TYPE_i] == 1,
        m_raw[..., SIZE_i],
        0
    ).cumsum(axis=-1)
    liq_taken = jnp.where(
        m_raw[..., EVENT_TYPE_i] > 1,
        m_raw[..., SIZE_i],
        0
    ).cumsum(axis=-1)
    return liq_prov, liq_taken

@jax.jit
@jax.vmap
def mid_price_loss_l1(
        gen_l2: jax.Array,
        act_l2: jax.Array,
    ) -> jax.Array:
    gen_mid = (gen_l2[0] + gen_l2[2]) / 2.
    act_mid = (act_l2[0] + act_l2[2]) / 2.
    return jnp.abs(gen_mid - act_mid) / 100
mid_price_loss_l1_batch = jax.vmap(mid_price_loss_l1, in_axes=(0, 0))

@jax.jit
def mid_price_ret_squ_err(
        # gen_l2: jax.Array,
        # act_l2: jax.Array,
        gen_mid: jax.Array,
        act_mid: jax.Array,
        mid_t0: jax.Array,
    ) -> jax.Array:
    # gen_mid = (gen_l2[:, 0] + gen_l2[:, 2]) / 2.
    # act_mid = (act_l2[:, 0] + act_l2[:, 2]) / 2.
    gen_ret = (gen_mid - mid_t0) / mid_t0
    act_ret = (act_mid - mid_t0) / mid_t0
    return jnp.square(gen_ret - act_ret)

@jax.jit
def union_price_repr(a, b):
    # append prices from b not already contained in a to a
    p_not_in_a = jnp.setdiff1d(
        b[:, 0],
        a[:, 0],
        #assume_unique=True,
        size=b.shape[0],
        fill_value=-1,
    ).reshape(-1, 1)
    a = jnp.concatenate(
        [a,
            jnp.concatenate([p_not_in_a, jnp.zeros_like(p_not_in_a)], axis=1)],
        axis=0
    )
    k, v = jax.lax.sort_key_val(a[:, 0], a[:, 1])
    a = jnp.concatenate(
        (k.reshape(-1, 1), v.reshape(-1, 1)),
        axis=1
    )
    return a



@partial(jax.jit, static_argnums=(2,))
def book_vol_comp(
        gen_l2: jax.Array,
        act_l2: jax.Array,
        n_price_levels: int,
    ):
    """
    """
    # step, (price, volume), level
    gen_l2 = gen_l2.reshape(-1, 2)#[:2 * n_price_levels]
    act_l2 = act_l2.reshape(-1, 2)#[:2 * n_price_levels]

    # take the union of n price levels for act and gen
    # --> used as cutoff for error calculation
    p_max_act = act_l2[: 2 * n_price_levels, 0].max()
    p_min_act = jnp.where(act_l2[:2 * n_price_levels, 0] != -1, act_l2[:2 * n_price_levels, 0], jnp.inf).min()
    p_max_gen = gen_l2[: 2 * n_price_levels, 0].max()
    p_min_gen = jnp.where(gen_l2[:2 * n_price_levels, 0] != -1, gen_l2[:2 * n_price_levels, 0], jnp.inf).min()
    p_max = jnp.max(jnp.array([p_max_act, p_max_gen]))
    p_min = jnp.min(jnp.array([p_min_act, p_min_gen]))
    # print(p_min, p_max)

    # turn ask volume negative
    gen_l2 = gen_l2.at[::2, 1].set(-gen_l2[::2, 1])
    act_l2 = act_l2.at[::2, 1].set(-act_l2[::2, 1])

    # p_max = act_l2[-2, 0]
    # p_min = act_l2[-1, 0]
    
    # sort by price
    gen_l2 = jnp.concatenate([gen_l2[-1: : -2], gen_l2[0: : 2]], axis=0)
    act_l2 = jnp.concatenate([act_l2[-1: : -2], act_l2[0: : 2]], axis=0)

    gen_l2_ = union_price_repr(gen_l2, act_l2)
    act_l2_ = union_price_repr(act_l2, gen_l2)

    # print('gen')
    # print(gen_l2_.shape)
    # print(gen_l2_)
    # print()
    # print('act')
    # print(act_l2_.shape)
    # print(act_l2_)
    
    # print(jnp.concatenate([gen_l2_, act_l2_], axis=1))

    p_sel_mask = (act_l2_[:, 0] >= p_min) & (act_l2_[:, 0] <= p_max)
    # print(p_sel_mask)
    vol_act = jnp.where(p_sel_mask, act_l2_[:, 1], 0)
    vol_gen = jnp.where(p_sel_mask, gen_l2_[:, 1], 0)
    n_levels = p_sel_mask.sum()

    return vol_gen, vol_act, n_levels

@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, 0, None))
def book_loss_l1(
        gen_l2: jax.Array,
        act_l2: jax.Array,
        n_price_levels: int,
    ) -> jax.Array:
    """ Calculates mean L1 loss between generated and actual book states (level 2 volume representations).
        The loss is summed for the union of n_price_levels from both books and divided by the number of
        price levels in the union.
    """
    vol_gen, vol_act, n_levels = book_vol_comp(gen_l2, act_l2, n_price_levels)
    # print(vol_act)
    # print(vol_gen)
    # sum of L1 losses, divided by number of price levels
    return jnp.abs(vol_act - vol_gen).sum() / (n_levels)

book_loss_l1_batch = jax.vmap(book_loss_l1, in_axes=(0, None, None))

@partial(jax.jit, static_argnums=(2,))
@partial(jax.vmap, in_axes=(0, 0, None))
def book_loss_wass(
        gen_l2: jax.Array,
        act_l2: jax.Array,
        n_price_levels: int,
    ):
    """ Calculates mean Wasserstein distance between generated and actual book states (level 2 volume representations).
        The distance is calculated for the union of n_price_levels from both books.
    """
    def running_sum(carry, x):
        s = carry + x
        return s, s

    vol_gen, vol_act, n_levels = book_vol_comp(gen_l2, act_l2, n_price_levels)
    # normalize
    vol_gen = vol_gen / jnp.sum(jnp.abs(vol_gen))
    vol_act = vol_act / jnp.sum(jnp.abs(vol_act))
    # print(jnp.vstack([vol_gen, vol_act]).T)
    # calculate wasserstein distance
    diff = vol_gen - vol_act
    # print(diff)
    _, ws_i = jax.lax.scan(running_sum, 0, diff)
    # print(ws_i)
    return jnp.sum(jnp.abs(ws_i))

book_loss_wass_batch = jax.vmap(book_loss_wass, in_axes=(0, None, None))

@jax.jit
@partial(jax.vmap, in_axes=(1, 1))
def return_corr(rets_gen, rets_eval):
    corr = jnp.corrcoef(rets_gen, rets_eval, rowvar=False)
    #return jnp.nan_to_num(corr[0, 1])
    return corr[0, 1]

@partial(jax.jit, static_argnums=(3,4))
def return_corr_se(rets_gen, rets_eval, rng, ci=0.95, n_bootstrap=1000):
    @partial(jax.vmap, in_axes=(None, 0))
    def bootstrap_sample(x, rng):
        return jax.random.choice(rng, x, shape=(x.shape[0],), replace=True)

    # vmap over repeated samples of rets_gen
    rngs = jax.random.split(rng, n_bootstrap)
    return_corr_ = jax.vmap(return_corr, in_axes=(0, 0))
    bs_gen, bs_eval = jnp.split(
        bootstrap_sample(
            jnp.concatenate([rets_gen, rets_eval], axis=1),
            rngs
        ),
        rets_gen.shape[1:2],
        axis=-1
    )
    #return bs_gen, bs_eval
    bootstr_data = return_corr_(bs_gen, bs_eval)
    #return bootstr_data
    return jnp.nanpercentile(
        bootstr_data,
        jnp.array([100 * (1-ci)/2, 100 * (ci + (1-ci)/2)]),
        axis=0
    )

@jax.jit
@partial(jax.vmap, in_axes=(None, 0))
def _event_type_count(
        event_types_data: jax.Array,
        event_type: int
    ) -> jax.Array:
    return jnp.sum(event_types_data == event_type)

@jax.jit
def event_type_count(
        event_types_data: jax.Array,
    ) -> jax.Array:

    @partial(jax.vmap, in_axes=(None, 0))
    def _event_type_count(
            event_types_data: jax.Array,
            event_type: jax.Array
        ) -> jax.Array:
        return jnp.sum(event_types_data == event_type)
    
    types = jnp.arange(1, 5)
    return _event_type_count(event_types_data, types)

def plot_order_type_frequency(event_types_gen, event_types_eval, combine_mod=False):
    """ plot frequency of order types over all runs (all samples and repeats)
    """
    width = 0.4
    if combine_mod:
        x = onp.array(range(1,4))
        event_types_gen = event_types_gen.at[..., 1].set(
            event_types_gen[..., 1] + event_types_gen[..., 2])
        event_types_eval = event_types_eval.at[..., 1].set(
            event_types_eval[..., 1] + event_types_eval[..., 2])
        event_types_gen = jnp.delete(event_types_gen, 2, axis=-1)
        event_types_eval = jnp.delete(event_types_eval, 2, axis=-1)
        labels = ('new order', 'canc / del', 'execution')
    else:
        x = onp.array(range(1,5))
        labels = ('new order', 'cancel', 'delete', 'execution')
        
    plt.bar(x-0.2, event_types_gen.sum(axis=(0, 1)), width=width, label='generated')
    plt.bar(x+0.2, event_types_eval.sum(axis=(0, 1)), width=width, label='eval')
    plt.xticks(x, labels)
    plt.legend()
    plt.title('Event Type Distribution')

def plot_log_hist(x, label=None):
    hist, bins = onp.histogram(x)
    logbins = onp.geomspace(x.min(), x.max(), 8)
    plt.hist(x, bins=logbins, label=label, alpha=0.5)
    plt.xscale('log')
    plt.legend()

def calc_moments(rets):
    r_mean = jnp.mean(rets, axis=0)
    r_var = jnp.var(rets, axis=0)
    r_skew = jnp.mean(jnp.power(rets, 3), axis=0) / jnp.power(r_var, 3/2)
    r_kurt = jnp.mean(jnp.power(rets, 4), axis=0) / jnp.power(r_var, 2) - 3
    return r_mean, r_var, r_skew, r_kurt

def plot_moments(
        r_gen_mean, r_gen_var, r_gen_skew, r_gen_kurt,
        r_eval_mean, r_eval_var, r_eval_skew, r_eval_kurt
    ):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(r_gen_mean, label='generated')
    axs[0, 0].plot(r_eval_mean, label='actual')
    axs[0, 0].set_title('Mean of Returns')
    axs[0, 0].set_xlabel('future messages')
    axs[0, 0].legend()

    axs[0, 1].plot(r_gen_var, label='generated')
    axs[0, 1].plot(r_eval_var, label='actual')
    axs[0, 1].set_title('Variance of Returns')
    axs[0, 1].set_xlabel('future messages')
    axs[0, 1].legend()

    axs[1, 0].plot(r_gen_skew, label='generated')
    axs[1, 0].plot(r_eval_skew, label='actual')
    axs[1, 0].set_title('Skewness of Returns')
    axs[1, 0].set_xlabel('future messages')
    axs[1, 0].legend()

    axs[1, 1].plot(r_gen_kurt, label='generated')
    axs[1, 1].plot(r_eval_kurt, label='actual')
    axs[1, 1].set_title('Kurtosis of Returns')
    axs[1, 1].set_xlabel('future messages')
    axs[1, 1].legend()

def plot_returns(rets_gen, rets_eval):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # share y-axis
    axs[0].plot(rets_gen.T)
    axs[0].set_title('Generated Returns')
    axs[0].set_xlabel('future messages')
    axs[0].set_ylabel('return')
    axs[1].plot(rets_eval.T)
    axs[1].set_title('Actual Returns')
    axs[1].set_xlabel('future messages')
    axs[1].set_ylabel('return')

def plot_ret_corr(rets_gen, rets_eval, ci=None, rng=None):
    if ci is not None:
        rng, rng_ = jax.random.split(rng)
        se = return_corr_se(
            rets_gen, rets_eval, rng_, n_bootstrap=10000)
        plt.fill_between(jnp.arange(rets_gen.shape[1]), se[0], se[1])

    ret_corr = return_corr(rets_gen, rets_eval)

    plt.plot(ret_corr, color='black')
    plt.title('n-step Correlation Coefficient of predicted returns')
    plt.xlabel('n-steps ahead')
    plt.ylabel('Correlation Coefficient')

def plot_book_losses(book_losses, book_losses_const, loss_type):
    plt.plot(
        book_losses.mean(axis=(0,1)),
        label='generated'
    )
    plt.plot(
        book_losses_const.mean(axis=0),
        label='fixed'
    )
    plt.legend()
    plt.title(f'Evolution of {loss_type} distance of L2 book volumes')
    plt.xlabel('future messages')
    plt.ylabel('mean abs. loss')

def plot_return_mse(ret_errs, ret_errs_const):
    plt.plot(
        jnp.sqrt(ret_errs.mean(axis=0)),
        label='generated'
    )
    plt.plot(
        jnp.sqrt(ret_errs_const.mean(axis=0)),
        label='fixed'
    )
    plt.legend()
    plt.title('MSE of mid price returns')
    plt.xlabel('Future Messages')
    plt.ylabel('Squared Error')

@jax.jit
def emp_cdf(x):
    x = jnp.sort(x)
    y = jnp.arange(1, len(x)+1) / len(x)
    return x, y

@jax.jit
def cdf_at(val, data):
    x, y = emp_cdf(data)
    return jnp.interp(val, x, y)

def prob_plot_2samples(gen, data):
    probs = jnp.arange(1, 100, 1)
    gen_pctile = jnp.percentile(gen, probs)
    y = cdf_at(gen_pctile, data)
    plt.plot(probs / 100, y, 'o')
    # plot 45 degree line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('Probability Plot')
    plt.xlabel('Generated')
    plt.ylabel('Data')
