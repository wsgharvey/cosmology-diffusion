"""
Train a diffusion model on images.
"""

import os
import sys
import argparse
import wandb
import torch.distributed as dist

from improved_diffusion import dist_util
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.logger import logger

os.environ["MY_WANDB_DIR"] = "none"
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"
    if "WANDB_DIR_DRYRUN" in os.environ:
        os.environ["MY_WANDB_DIR"] = os.environ["WANDB_DIR_DRYRUN"]


def init_wandb(config, id):
    if dist.get_rank() != 0:
        return
    wandb_dir = os.environ.get("MY_WANDB_DIR", "none")
    if wandb_dir == "none":
        wandb_dir = None
    wandb.init(entity=os.environ['WANDB_ENTITY'],
               project=os.environ['WANDB_PROJECT'],
               config=config, dir=wandb_dir, id=id)
    print(f"Wandb run id: {wandb.run.id}")
    num_nodes = 1
    if "SLURM_JOB_NODELIST" in os.environ:
        assert "SLURM_JOB_NUM_NODES" in os.environ
        num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        print(f"Node list: {os.environ['SLURM_JOB_NODELIST']}")
    logger.logkv("num_nodes", num_nodes)
    print(f"Number of nodes: {num_nodes}") 


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    resume = bool(args.resume_id)
    init_wandb(config=args, id=args.resume_id if resume else None)

    print("creating data loader...")
    data = load_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        single_data_point=args.single_data_point,
    )
    if args.image_size is None:
        args.image_size = next(data)[0].shape[-1]

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_interval=args.sample_interval,
        args=args,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        resume_id='',  # set this to a previous run's wandb id to resume training
        sample_interval=50000,
        single_data_point=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
