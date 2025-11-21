import numpy as np
import jax.numpy as jnp
from loaderx import NPDataset,DataLoader

def transform(data, label):
    return (data[..., None] / 255.0).astype(jnp.bfloat16) , label.astype(np.int32)

def loader(data, label):
  return DataLoader(NPDataset(data), NPDataset(label),transform=transform)