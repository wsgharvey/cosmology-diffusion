import os
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th


files = [
    ('Data_005_density.dat', 1., 0., 1.),
    ('Data_004_density.dat', 2/3, 0.5, 0.77318105778234214),
    ('Data_003_density.dat', 1/2, 1., 0.61180631557600529),
    ('Data_002_density.dat', 1/3, 2., 0.42144646268973707),
    ('Data_001_density.dat', 1/4, 3., 0.31884106654994515),
    ('Data_000_density.dat', 1/5, 4., 0.25588245948185767),
]


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


def load_data(data_path, batch_size, image_channels, max_data_value, single_data_point=False, deterministic=False):

    def load_data(fname):
        array = np.loadtxt(os.path.join(data_path, fname)).astype(np.float32)
        image_size_cubed = len(array[:, 3])
        image_size = int(np.cbrt(image_size_cubed))
        array = array[:, 3].reshape((image_size, image_size, image_size))
        array = -1 + np.log(2 + array)
        assert max_data_value >= array.max()
        return array
    all_data = [(load_data(path), a, z, g) for path, a, z, g, in files]
    batch = []
    ys = []
    while True:
        # randomly sample slice of our data
        idx = np.random.randint(0, len(all_data))
        data, _, _, g = all_data[idx]
        slice_dim = np.random.randint(0, 3)
        image_size = data.shape[slice_dim]
        slice_i = np.random.randint(0, image_size)
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
        ys.append(g)
        if len(batch) == batch_size:
            yield th.stack(batch), {'y': th.tensor(ys).view(-1, 1)}
            batch = []
            ys = []


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