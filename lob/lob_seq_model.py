from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
#from ..s5.layers import SequenceLayer
from s5.seq_model import StackedEncoderModel, masked_meanpool


class LobPredModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    output_dims: Tuple[int]  # dimensions of different output classes
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                                        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

# Here we call vmap to parallelize across a batch of input sequences
BatchLobPredModel = nn.vmap(
    LobPredModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')
