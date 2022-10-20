from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import glob
from collections import defaultdict
import argparse
import pickle
import torch as th
import matplotlib.pyplot as plt

from image_sample import kwargs_fname


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate.")
    args = parser.parse_args()

    # load model samples
    files = sorted(glob.glob(str(Path(args.eval_dir) / "samples" / "*.npy")))[:args.n_samples]
    samples = np.stack([np.load(f) for f in files])
    print(samples.shape, samples.min(), samples.max())
    # load dataset samples (getting arguments from the saved config)
    model_config_path = Path(args.eval_dir) / "model_config.json"
    assert model_config_path.exists(), f"Could not find model config at {model_config_path}"
    with open(model_config_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    max_density = 5.5
    overdensities = samples.squeeze(axis=1)
    densities = np.exp(1 + overdensities) - 1  # densities in range [0, inf]

    model_kwargses = [pickle.load(open(kwargs_fname(f), 'rb')) for f in files]
    model_kwargses = [{k: v.detach().cpu().numpy() for k, v in m.items()} for m in model_kwargses]

    for sample_i, (sample, model_kwargs) in enumerate(zip(overdensities, model_kwargses)):
        D =  sample.shape[-1]
        indices = [0, 1, 2, 3, 4] + [D//2, D//2+1, D//2+2, D//2+3, D//2+4]
        fig, axes = plt.subplots(ncols=D, figsize=(D*2, 2))
        axes[0].set_ylabel(model_kwargs['y'].item())
        for ind, ax in zip(indices, axes):
            im = ax.imshow(sample[ind], cmap='gray')
            ax.set_title(str(ind))
            ax.axis('off')
            ax.set_ylim(-1, max_density)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        save_dir = Path(args.eval_dir) / "vis"
        save_dir.mkdir(parents=True, exist_ok=True) 
        fig.savefig(save_dir / f"sample_{sample_i}.pdf", bbox_inches='tight')
