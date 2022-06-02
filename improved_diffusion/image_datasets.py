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



# def load_data(
#     *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.

#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.

#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     """
#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#     )
#     if deterministic:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
#         )
#     while True:
#         yield from loader