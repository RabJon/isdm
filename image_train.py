"""
Train a diffusion model on images.
"""

import os
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data, load_data_from_file_paths
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch


def train(args):
    dist_util.setup_dist()
    
    if "log_dir" in vars(args):
        logger.configure(args.log_dir)
    else:
        logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    torch.cuda.empty_cache() 

    if args.train_file_paths and args.val_file_paths:
        logger.log("creating data loader with train and validation files...")
        logger.log("batch_size:" + str(args.batch_size))
        train_data = load_data_from_file_paths(
            dataset_mode=args.dataset_mode,
            file_paths=args.train_file_paths,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            random_crop=True,
            random_flip=True,
            deterministic=False
        )

        val_data = load_data_from_file_paths(
            dataset_mode=args.dataset_mode,
            file_paths=args.val_file_paths,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            random_crop=False,
            random_flip=False,
            deterministic=True
        )

        print("CUDA available 2:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
        logger.log("training...")
        torch.cuda.empty_cache() 
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data= train_data,
            val_data = val_data,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            drop_rate=args.drop_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            max_epochs = 20000
        ).run_loop()
    
    
    elif args.data_dir: #legacy mode without epochs and without validation set
        logger.log("creating data loader from data_dir...")
        logger.log("batch_size:" + str(args.batch_size))
        data = load_data(
            dataset_mode=args.dataset_mode,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            is_train=args.is_train
        )
        print("CUDA available 2:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            drop_rate=args.drop_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
        ).run_loop()
    else:
        raise ValueError("Either data_dir or train_file_paths and val_file_paths must be specified!")





def main():
    print("CUDA available 1:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count()) #this line makes the difference if CUDA is correctly detected or not
    args = create_argparser().parse_args()
    train(args)

    

def get_training_defaults():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True
    )

    return defaults



def create_argparser():
    defaults = get_training_defaults()
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
