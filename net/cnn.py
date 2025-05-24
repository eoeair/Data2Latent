# The Flax NNX API.
from flax import nnx
from functools import partial

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, rngs: nnx.Rngs, return_latent=True):
    self.return_latent = return_latent
      
    self.conv1 = nnx.Conv(1, 4, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(4, 2, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.embed = nnx.Linear(392, 2, rngs=rngs)
    self.logits = nnx.Linear(2, 10, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.conv1(x))
    x = nnx.relu(self.conv2(x))
    x = self.avg_pool(x)
    x = x.reshape(x.shape[0], -1)  # flatten
    latent = self.embed(x)
    logits = self.logits(latent)
    if self.return_latent:
        return latent, logits
    else:
        return logits