import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

class KadidDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.dataset = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_name = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
    image = io.imread(img_name)

    # Grayscale
    # image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    dmos_value = self.dataset.iloc[idx, 2] / max(self.dataset.iloc[:, 2])

    sample = [image, dmos_value]

    if self.transform:
      image = self.transform(image)
      sample = [image, dmos_value]

    return sample