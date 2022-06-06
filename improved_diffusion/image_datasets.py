from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th


def expand_channels(img, image_channels, max_val):
    if image_channels == 1:
        return img
    elif image_channels == 2:
        c1 = img.clamp(min=-1, max=1)
        c2 = img.clamp(min=1, max=max_val)
        c2 = (c2-1) / (max_val-1) * 2 - 1  # normalise to [-1, 1]
        return th.cat([c1, c2], dim=0)
    else:
        raise ValueError("image_channels must be 1 or 2")


def load_data(data_path, batch_size, image_size, image_channels, max_data_value, single_data_point=False, deterministic=False):
    data = np.loadtxt(data_path).astype(np.float32)
    data = data[:, 3].reshape((image_size, image_size, image_size))
    data = -1 + np.log(2 + data)
    assert max_data_value >= data.max()
    batch = []
    while True:
        # randomly sample slice of our data
        slice_i = np.random.randint(0, image_size)
        slice_dim = np.random.randint(0, 3)
        flip = np.random.randint(0, 2)
        rotation = np.random.randint(0, 4)
        if single_data_point:
            slice_i = slice_dim = flip = rotation = 0
        img = data.take(indices=slice_i, axis=slice_dim)
        if flip:
            img = img[::-1, :]
        img = np.rot90(img, k=rotation)
        img = th.tensor(img.copy()).view(1, image_size, image_size)
        img = expand_channels(img, image_channels, max_data_value)
        batch.append(img)
        if len(batch) == batch_size:
            yield th.stack(batch), {}
            batch = []


def collapse_two_channel(t, max_data_value):
    """ map Bx2xHxW to Bx1xHxW in range [-1, inf]
    """
    # clamp to valid range
    c1 = t[:, 0:1].clamp(min=-1, max=1)
    c2 = t[:, 1:2].clamp(min=-1, max=1)
    # impose additional constraint that one of c1 and c2 is at clamped region
    d1 = (1-c1).abs()
    d2 = (-1-c2).abs()
    c1[d1 < d2] = 1
    c2[d1 >= d2] = -1
    return c1 + (1 + c2) * (max_data_value - 1) / 2