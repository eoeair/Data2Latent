import numpy as np
from functools import partial
from loaderx import Dataset,DataLoader

def transform(batch, dtype):
    return (batch[0][..., None] / 255.0).astype(dtype) , batch[1].astype(np.int32)

def loader(data, label, dtype=np.float32):
  return DataLoader(Dataset(data, label), transform=partial(transform, dtype=dtype))