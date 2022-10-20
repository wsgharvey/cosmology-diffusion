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

from .image_sample import kwargs_fname


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

    densities = np.exp(1 + samples) - 1  # densities in range [0, inf]

    normed = densities

    model_kwargs = [pickle.load(open(kwargs_fname(f), 'rb')) for f in files]
    model_kwargs = {k: th.stack([m[k] for m in model_kwargs]) for k in model_kwargs[0].keys()}
    model_kwargs = {k: v.detach().cpu().numpy() for k, v in model_kwargs.items()}

    print(samples[0].shape, samples[0].min(), samples[0].max())
    print(model_kwargs['y'])

    # fig.savefig(Path(args.eval_dir) / f"density-histogram-{args.n_samples}.pdf", bbox_inches='tight')
