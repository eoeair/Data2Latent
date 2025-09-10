from flax import nnx
import jax.numpy as jnp

class CNN(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dtype=jnp.bfloat16):
    self.dtype = dtype
    
    self.conv1 = nnx.Conv(1, 8, kernel_size=(9, 9), padding="SAME", rngs=rngs, dtype=dtype)
    self.conv2 = nnx.Conv(8, 16, kernel_size=(9, 9), padding="SAME", rngs=rngs, dtype=dtype)
    self.bn1 = nnx.BatchNorm(8, rngs=rngs, dtype=jnp.float32)
    self.bn2 = nnx.BatchNorm(16, rngs=rngs, dtype=jnp.float32)

    self.conv1 = nnx.Sequential(self.conv1, self.bn1, nnx.relu)
    self.conv2 = nnx.Sequential(self.conv2, self.bn2, nnx.relu)

    self.embed = nnx.Linear(16, 2, rngs=rngs, dtype=jnp.float32)
    self.logits = nnx.Linear(2, 10, rngs=rngs, dtype=jnp.float32)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.mean(axis=(1, 2))
    latent = self.embed(x)
    logits = self.logits(latent)
    return latent, logits