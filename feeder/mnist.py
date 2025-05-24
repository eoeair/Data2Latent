import numpy as np
import loaderx

class Mnist(loaderx.Dataset):
  def __init__(self, dataset_path, data, label):
    self._data = np.load(dataset_path)[data]
    self._label = np.load(dataset_path)[label]

  def __getitem__(self, idx):
    # expand axis(-1)
    return self._data[idx, ..., None]/np.float32(255), self._label[idx].astype(np.int32)
  def __len__(self):
    return len(self._data)

def loader(dataset_path, data, label, batch_size, num_epoch):
  return loaderx.DataLoader(dataset = Mnist(dataset_path, data, label),batch_size=batch_size,num_epoch=num_epoch)
