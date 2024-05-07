import jax.numpy as jnp
from jax import random
from flax.core.frozen_dict import FrozenDict
import torch
import numpy as np
# Example JAX model parameters (using FrozenDict)
"""jax_params = FrozenDict({"layer1": {"weights": random.normal(random.PRNGKey(0), (2, 3)),
                                    "biases": random.normal(random.PRNGKey(1), (3,))},
                         "layer2": {"weights": random.normal(random.PRNGKey(2), (3, 2)),
                                    "biases": random.normal(random.PRNGKey(3), (2,))}})
def frozen_dict_to_pytorch(frozen_dict):
    '''
    Recursively converts JAX model parameters (FrozenDict) into PyTorch model parameters.
    '''
    if isinstance(frozen_dict, FrozenDict):
        return {k: frozen_dict_to_pytorch(v) for k, v in frozen_dict.items()}
    else:
        # Convert numpy arrays to PyTorch tensors
        np_array = np.asarray(frozen_dict)
        return torch.nn.Parameter(torch.from_numpy(np_array))
    
print(jax_params)

pytorch_params = frozen_dict_to_pytorch(jax_params)
# Example: Accessing PyTorch parameters for layer1 weights
print(pytorch_params['layer1']['weights'])"""


a = np.array(['a','b','c'])


print(a=='b')
