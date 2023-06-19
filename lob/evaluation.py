import jax
import jax.numpy as jnp
from functools import partial


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

@partial(jax.vmap, in_axes=(1, 1))
def return_corr(rets_gen, rets_eval):
    corr = jnp.corrcoef(rets_gen, rets_eval, rowvar=False)
    return corr[0, 1]

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
            event_type: int
        ) -> jax.Array:
        return jnp.sum(event_types_data == event_type)
    
    types = jnp.arange(1, 5)
    return _event_type_count(event_types_data, types)