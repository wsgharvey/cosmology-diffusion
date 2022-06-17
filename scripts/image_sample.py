"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import json

import numpy as np
import torch as th

from improved_diffusion import dist_util  # we do NOT support distributed sampling
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    str2bool,
)
from improved_diffusion.test_util import get_model_results_path, Protect


def main(model, args):

    def fname(saved):
        return args.eval_dir / "samples" / f"sample-{saved:06d}.npy"

    print("sampling...")
    saved = 0
    while saved < args.n_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.image_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample.squeeze(dim=1).contiguous()

        for img in sample.cpu().numpy():
            np.save(fname(saved), img)
            while os.path.exists(fname(saved)):
                saved += 1

    print("sampling complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples desired.")
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Prepare samples directory
    args.eval_dir = get_model_results_path(args)
    (args.eval_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_args.update({"use_ddim": args.use_ddim,
                       "timestep_respacing": args.timestep_respacing})
    model_args = argparse.Namespace(**model_args)
    # Load the model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    args.image_size = model_args.image_size
    args.image_channels = model_args.image_channels

    # write config dictionary to the results directory
    json_path = args.eval_dir / "model_config.json"
    if not json_path.exists():
        with Protect(json_path): # avoids race conditions
            with open(json_path, "w") as f:
                json.dump(vars(model_args), f, indent=4)
        print(f"Saved model config at {json_path}")

    main(model, args)
