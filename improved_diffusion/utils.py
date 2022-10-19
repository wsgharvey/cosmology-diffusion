import torch as th


def concat_images_with_padding(images, horizontal=True,
                               pad_dim=1, pad_val=0, pad_ends=False):
    """Cocatenates a list (or batched tensor) of CxHxW images, with padding in
    between, for pretty viewing.
    """
    *_, h, w = images[0].shape
    pad_h, pad_w = (h, pad_dim) if horizontal else (pad_dim, w)
    padding = th.zeros_like(images[0][..., :pad_h, :pad_w]) + pad_val
    images_with_padding = []
    for image in images:
        images_with_padding.extend([image, padding])
    if pad_ends:
        images_with_padding = [padding, *images_with_padding, padding]
    images_with_padding = images_with_padding[:-1]   # remove final pad
    return th.cat(images_with_padding, dim=-1 if horizontal else -2)