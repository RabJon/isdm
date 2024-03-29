"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os


from guided_diffusion.image_datasets import load_data, load_data_from_file_paths

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import torch as th
import torch.distributed as dist
import torchvision as tv



def sample(args):
    dist_util.setup_dist()
    
    if "log_dir" in vars(args):
        logger.configure(args.log_dir)
    else:
        logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")

    if args.file_paths:
        data = load_data_from_file_paths(
            dataset_mode=args.dataset_mode,
            file_paths=args.file_paths,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            random_crop=False,
            random_flip=False,
            deterministic=True
        )
    else:
        data = load_data(
            dataset_mode=args.dataset_mode,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            random_crop=False,
            random_flip=False,
            is_train=True #orig is False: original uses Test set for sampling, which is not what I want ;(
        )

    # data = load_data(
    #     dataset_mode=args.dataset_mode,
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    #     deterministic=True,
    #     random_crop=False,
    #     random_flip=False,
    #     is_train=True #orig is False: original uses Test set for sampling, which is not what I want ;(
    # )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    if args.num_classes > 2:
        import numpy as np
        import skimage.io as io
        colored_labels_path = os.path.join(args.results_path, 'colored_labels')
        os.makedirs(colored_labels_path, exist_ok=True)
        mask_colors = np.array([[0,0,0],[1,0,103],[213,255,0],[255,0,86],[158,0,142],[14,76,161],[255,229,2],[0,95,57],[0,255,0],[149,0,58],[255,147,126],[164,36,0],
            [0,21,68],[145,208,203],[98,14,0],[107,104,130],[0,0,255],[0,125,181],[106,130,108],[0,174,126],[194,140,159],[190,153,112],[0,143,156],
            [95,173,78],[255,0,0],[255,0,246],[255,2,157],[104,61,59],[255,116,163],[150,138,232],[152,255,82],[167,87,64],[1,255,254],[255,238,232],
            [254,137,0],[189,198,255],[1,208,255],[187,136,0],[117,68,177],[165,255,210],[255,166,254],[119,77,0],[122,71,130],[38,52,0],[0,71,84],
            [67,0,44],[181,0,255],[255,177,103],[255,219,102],[144,251,146],[126,45,210],[189,211,147],[229,111,254],[222,255,116],[0,255,120],
            [0,155,255],[0,100,1],[0,118,255],[133,169,0],[0,185,23],[120,130,49],[0,255,198],[255,110,65],[232,94,190]] + [[255,255,255]] * 192, dtype=np.uint8)



    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0).cuda()
        label = (cond['label_ori'].float() / 255.0).cuda()
            
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, image.shape[2], image.shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            # tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            # tv.utils.save_image(sample[j], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            # tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(i) +'.png'))
            tv.utils.save_image(sample[j], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(i) + '.png'))
            tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(i) + '.png'))
            if args.num_classes > 2:
                colored_label = mask_colors[cond['label_ori'][j].cpu().numpy()]
                save_path = os.path.join(colored_labels_path, cond['path'][j].split('/')[-1].split('.')[0] + "_" + str(i) + '.png')
                io.imsave(save_path, colored_label, check_contrast=False)



        logger.log(f"created {len(all_samples) * args.batch_size} samples")

        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")


def main():
    print("CUDA available 1:", th.cuda.is_available()) #this line makes the difference if CUDA is correctly detected or not
    args = create_argparser().parse_args()
    sample(args)

    


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    # data['label'] = data['label'].long()
    data['label'] = data['label'].cuda().long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    # input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_label = th.cuda.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def get_sampling_defaults():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0
    )
    return defaults

def create_argparser():
    defaults = get_sampling_defaults()
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
