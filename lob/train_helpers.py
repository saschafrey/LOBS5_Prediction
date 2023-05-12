from functools import partial
import numpy as onp
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Tuple

from lob.lob_seq_model import LobPredModel


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       use_book_data,
                       book_dim,
                       book_seq_len,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False,
                       num_devices=1,
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    # batch size is given for data across all devices
    # i.e. batch is split between GPUs but dummy data is per GPU
    assert bsz % num_devices == 0
    bsz = bsz // num_devices

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        if use_book_data:
            dummy_input = (
                np.ones((bsz, seq_len, in_dim)),
                np.ones((bsz, book_seq_len, book_dim)),
            )
            integration_timesteps = (
                np.ones((bsz, seq_len, )),
                np.ones((bsz, book_seq_len, )),
            )
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)) , )
            integration_timesteps = (np.ones((bsz, seq_len, )), )

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           *dummy_input, *integration_timesteps,
                           )
    if batchnorm:
        params = variables["params"].unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"].unfreeze()
        # Note: `unfreeze()` is for using Optax.

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    #print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_slices(dims):
    slices = []
    last_i = 0
    for d in dims:
        slices.append(slice(last_i, last_i+d))
        last_i += d
    return slices

# Train and eval steps
#@jax.jit
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])
    return -np.sum(one_hot_label * logits)
    #return -np.sum(label * logits)


'''
def mse_loss(preds, targets):
    return np.sum((preds - targets)**2)


def rmse_loss(preds, targets):
    return np.sqrt(mse_loss(preds, targets))


def weighted_loss(
    outputs,
    targets,
    loss_fns=[cross_entropy_loss, cross_entropy_loss, mse_loss],
    weights=[0.25, 0.25, 0.5],
    dims=[2, 7, 6]
):

    assert np.sum(dims) == outputs.shape[1]
    assert np.sum(weights) == 1.

    slices = get_slices(dims)
    return np.sum(
        [w * loss_fn(outputs[s], targets[s]) for w, loss_fn, s in zip(weights, loss_fns, slices)])
'''

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

#@jax.jit
#def compute_accuracy(logits, dummy_labels):
#    bool_idx = np.arange(logits.shape[0]), np.argmax(logits, axis=1)
#    return dummy_labels[bool_idx] == 1

'''
def weighted_accuracy_and_rmse(
    outputs,
    targets,
    acc_weights=[0.5, 0.5],
    dims=[2, 7, 6]
):
    slices = get_slices(dims)
    acc = np.sum(
        [w * compute_accuracy(outputs[s], targets[s]) for w, s in zip(acc_weights, slices[:-1])])
    rmse = rmse_loss(outputs[slices[-1]], targets[slices[-1]])
    return acc, rmse
'''

def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[Tuple, np.ndarray, Tuple]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    assert inputs.shape[1] == seq_len

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps_msg' in aux_data:
        integration_timesteps = (np.diff(aux_data['timesteps_msg']), )
    else:
        integration_timesteps = (np.ones((len(inputs), seq_len)), )

    if "book_data" in aux_data:
        full_inputs = (inputs.astype(float), aux_data['book_data'])
        if "timesteps_book" in aux_data:
            integration_timesteps += (np.diff(aux_data['timesteps_book']), )
        else:
            integration_timesteps += (np.ones((len(inputs), seq_len)), )
    else:
        full_inputs = (inputs.astype(float), )

    return full_inputs, np.squeeze(targets.astype(float)), integration_timesteps


def device_reshape(
        inputs: Tuple,
        targets: np.ndarray,
        integration_timesteps: Tuple,
        n_devices: int
    ) -> Tuple:
    """
    Reshape inputs, targets, and integration timesteps for multi-device training.
    :param inputs:                  (tuple) inputs as returned from prep_batch.
    :param targets:                 (np.ndarray) targets as returned from prep_batch.
    :param integration_timesteps:   (tuple) integration timesteps as returned from prep_batch.
    :param n_devices:               (int) number of devices.
    :return:
    """
    inputs = tuple([np.reshape(x, (n_devices, -1, *x.shape[1:])) for x in inputs])
    targets = np.reshape(targets, (n_devices, -1, *targets.shape[1:]))
    integration_timesteps = tuple(
        [np.reshape(x, (n_devices, -1, *x.shape[1:])) for x in integration_timesteps]
    )

    return inputs, targets, integration_timesteps



def train_epoch(
        state,
        rng,
        model,
        trainloader,
        seq_len,
        in_dim,
        batchnorm,
        lr_params,
        num_devices,
    ):
    """
    Training function for an epoch that loops over batches.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)

        #if num_devices > 1:
        inputs, labels, integration_times = device_reshape(
            inputs, labels, integration_times, num_devices)

        #print(inputs.shape)
        #print(labels.shape)
        #print(integration_times.shape)
        #import sys
        #sys.exit()

        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            num_devices,
        )
        print('finished train step')
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def validate(state, model, testloader, seq_len, in_dim, batchnorm, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    #split_indices = tuple(onp.cumsum(onp.array(model.output_dims))[:-1])
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
        loss, acc, pred = eval_step(
            inputs, labels, integration_timesteps, state, model, batchnorm)
        losses = np.append(losses, loss)
        accuracies = np.append(accuracies, acc)

    aveloss, aveaccu = np.mean(losses), np.mean(accuracies)
    return aveloss, aveaccu


@partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(
        state,
        rng,
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        model,
        batchnorm,
        num_devices,
    ):
    """ Performs a single training step given a batch of data
        NOTE: batch_inputs is a tuple of (batched_message_inputs, batched_book_inputs)
              or only (batched_message_inputs,) if book inputs are not used.
    """
    # def loss_fn(params):

    #     if batchnorm:
    #         logits, mod_vars = state.apply_fn( #model.apply(
    #             {"params": params, "batch_stats": state.batch_stats},
    #             *batch_inputs, *batch_integration_timesteps,
    #             rngs={"dropout": rng},
    #             mutable=["intermediates", "batch_stats"],
    #         )
    #     else:
    #         logits, mod_vars = state.apply_fn( # model.apply(
    #             {"params": params},
    #             *batch_inputs, *batch_integration_timesteps,
    #             rngs={"dropout": rng},
    #             mutable=["intermediates"],
    #         )

    #     # average cross-ent loss
    #     loss = np.mean(cross_entropy_loss(logits, batch_labels))
        
    #     # not necessary if labels are already one-hot:
    #     '''
    #     cum_dims = np.cumsum(model.output_dims)
    #     loss = np.sum(
    #         np.mean(cross_entropy_loss(log, lab))
    #         for log, lab in zip(
    #             np.split(logits, cum_dims[:-1], axis=1),
    #             np.split(batch_labels, cum_dims[:-1], axis=1)
    #         )
    #     )
    #     '''

    #     return loss, (mod_vars, logits)

    # (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # if num_devices > 1:
    #     fpass = jax.pmap(
    #         forward_pass,
    #         axis_name="batch_devices",
    #         static_broadcasted_argnums=(1, 7),
    #         in_axes=(None, None, None, None, 0, 0, 0, None))
    # else:
    # fpass = forward_pass

    # print('len(batch_inputs)', len(batch_inputs))
    # print('batch_inputs[0].shape', batch_inputs[0].shape)
    # print('batch_inputs[1].shape', batch_inputs[1].shape)

    loss, mod_vars, grads = forward_pass(
        state.params,
        state.apply_fn,
        state.batch_stats,
        rng,
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        batchnorm
    )
    # calculate means over device dimension (first)
    loss = loss.mean()
    mod_vars = jax.tree_map(lambda x: x.mean(axis=0), mod_vars)
    grads = jax.tree_map(lambda x: x.mean(axis=0), grads)

    # print('loss.shape', loss.shape)
    # print('grad shapes:')
    # jax.tree_map(lambda x: print(x.shape), grads)
    # print()

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(1, 7),
    in_axes=(None, None, None, None, 0, 0, 0, None))
def forward_pass(
        params,  # 0
        apply_fn,  # 1
        batch_stats,  # 2
        rng,  # 3
        batch_inputs, # 4
        batch_labels, # 5
        batch_integration_timesteps, # 6
        batchnorm, # 7
    ):
    def loss_fn(params):

        # print('in loss_fn')
        # print('params["message_encoder"]["encoder"]["kernel"].shape', 
        #       params["message_encoder"]["encoder"]["kernel"].shape)
        # print('batch_inputs[0].shape', batch_inputs[0].shape)
        # print('batch_integration_timesteps[0].shape', batch_integration_timesteps[0].shape)
        # print('batch_labels.shape', batch_labels.shape)
        # print('param shapes:')
        # jax.tree_map(lambda x: print(x.shape), params)
        # print()

        if batchnorm:
            logits, mod_vars = apply_fn( # state.apply_fn( # model.apply(
                {"params": params, "batch_stats": batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = apply_fn( # state.apply_fn( # model.apply(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # average cross-ent loss
        loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    return loss, mod_vars, grads

@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs,
              batch_labels,
              batch_integration_timesteps,
              state,
              model,
              #split_indices,
              batchnorm,
              ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             *batch_inputs, *batch_integration_timesteps,
                             )
    else:
        logits = model.apply({"params": state.params},
                             *batch_inputs, *batch_integration_timesteps,
                             )

    #losses = cross_entropy_loss(logits, batch_labels) / (len(split_indices) + 1)
    #losses = weighted_loss(logits, batch_labels)
    losses = cross_entropy_loss(logits, batch_labels)
    
    accs = compute_accuracy(logits, batch_labels)
    #accs = weighted_accuracy_and_rmse(logits, batch_labels)

    # calculate mean accuracy per classification task (e.g. price, order_type etc)
    #accs = np.mean(  # mean over different classification tasks
    #        np.array([
    #            compute_accuracy(log, lab) for log, lab in zip(
    #                np.split(logits, split_indices, axis=1),
    #                np.split(batch_labels, split_indices, axis=1))
    #        ]),
    #    axis=0
    #)

    return losses, accs, logits
