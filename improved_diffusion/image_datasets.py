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
start_file = 'start_density.dat'


def load_data(data_path, batch_size, single_data_point=False, deterministic=False, density_3D=False):

    def load_data(fname, normalize=True):
        array = np.loadtxt(os.path.join(data_path, fname)).astype(np.float32)
        image_size_cubed = len(array[:, 3])
        image_size = int(np.cbrt(image_size_cubed))
        array = array[:, 3].reshape((image_size, image_size, image_size))
        return -1 + np.log(2 + array) if normalize else array
    all_data = [(load_data(fname), a, z, g) for fname, a, z, g, in files]
    start_data = load_data(start_file, normalize=False)
    start_data = (start_data - start_data.mean()) / start_data.std()
    batch = []
    ys = []
    while True:
        # randomly sample slice of our data
        idx = np.random.randint(0, len(all_data)) # MEAD: TODO: random.choice()
        data, _, _, g = all_data[idx]
        image_size = data.shape[0] # MEAD: Added
        if density_3D: # MEAD: Added
            flips = np.random.randint(0, 2, size=3) # MEAD: TODO: Code sharing with 2D case
            rotations = np.random.randint(0, 4, size=3)
            if single_data_point:
                flips = np.zeros(3, dtype=np.int32)
                rotations = np.zero(4, dtype=np.int32)
            def get_field(array, flips, rotations, image_size):
                for n_rotate, axes in zip(rotations, [(0, 1), (0, 2), (1, 2)]):
                    array = np.rot90(array, k=n_rotate, axes=axes)
                for axis, do_flip in enumerate(flips):
                    array = np.flip(array, axis=axis)
                return th.tensor(array.copy()).view(1, image_size, image_size, image_size)
        else:
            slice_dim = np.random.randint(0, 3)
            #image_size = data.shape[slice_dim]
            slice_i = np.random.randint(0, image_size)
            flip = np.random.randint(0, 2)
            rotation = np.random.randint(0, 4)
            if single_data_point:
                slice_i = slice_dim = flip = rotation = 0
            def get_slice(array, slice_i, slice_dim, flip, rotation):
                img = array.take(indices=slice_i, axis=slice_dim)
                if flip:
                    img = img[::-1, :]
                img = np.rot90(img, k=rotation)
                return th.tensor(img.copy()).view(1, image_size, image_size)
        if density_3D: # MEAD: Added
            img = get_field(data, flips, rotations, image_size)
            start = get_field(start_data, flips, rotations, image_size)
        else:
            img = get_slice(data, slice_i, slice_dim, flip, rotation)
            start = get_slice(start_data, slice_i, slice_dim, flip, rotation)
        batch.append(img)
        ys.append({'y': th.tensor(g).view(1), 'image_cond': start})
        if len(batch) == batch_size:
            yield th.stack(batch), {k: th.stack([y[k] for y in ys]) for k in ys[0]}
            batch = []
            ys = []