# Improved Semantic Diffusion Model (ISDM)

A method to enhance existing supervised (and scarce) semantic segmentation datasets for defect/anomaly detection with synthetic data from Semantic Diffusion Models.

&nbsp;

<img src='assets\sampling.png' align="left">  

&nbsp;

## Paper
This project was part of my master thesis on [Improving Semantic Segmentation Models through Synthetic Data Generation via Diffusion Models](./assets/Master_Thesis.pdf). 

## Abstract
Industrial datasets for semantic segmentation are typically scarce because of the time-consuming annotation procedure. However, modern deep learning model architectures for this task require a large amount of data and therefore perform below their potential capabilities when trained on small datasets. This makes semantic segmentation on industrial datasets a suitable candidate for the application of current advances with generative models. Particularly, recent Diffusion Models (DMs), which are mainly known for applications in the field of art, can be exploited to enhance existing datasets with synthetic data. In principle, DMs can be used to sample arbitrary amounts of data and thus to adapt the sparse existing datasets such that the models for semantic segmentation can be better trained on them. In the case of semantic segmentation, however, it is not enough just to generate the images, the images must also matchwith corresponding labels on a pixel level. In this thesis, two main approaches, namely semantic imagesynthesis and paired image-mask synthesis, are discussed in detail. Furthermore, two implementations for these approaches are proposed that allow training DMs on small-scale industrial semantic segmentation datasets and generating corresponding synthetic datasets. These implementations are referred to as “All-Latent” and “Improved Semantic Diffusion Model” (ISDM), named after the types of DMs on which they are built. Extensive experiments with three different semantic segmentation architectures are conducted to evaluate the impact of the synthetic datasets generated by All-Latent and ISDM in terms of the mean intersection-over-union (mIoU) score. They show that for both methods the generated data can contribute to improved performance of the downstream semantic segmentation models without modifying the models themselves. Especially when the models are trained on a mix of real and synthetic data, performance improvements with respect to the baseline model, which is trained on real data only, can be achieved in several experiments. This could be demonstrated for binary as well as for multi-class segmentation. For some experiments, it was even possible to achieve improvements when training the models only with synthetic data.


## Acknowledgement
This repository is strongly based on the work of [semantic-diffusion-model](https://github.com/WeilunWang/semantic-diffusion-model) which in turn builds on [guided-diffusion](https://github.com/openai/guided-diffusion). The evaluation is taken from [conditional-polyp-diffusion](https://github.com/simulamet-host/conditional-polyp-diffusion). The dataset on which the approach was predominantly tested is the [Lemons quality control dataset](https://github.com/softwaremill/lemon-dataset).


## Evaluation
asdf

For further information about the evaluation strategy and the obtained results please refer to my [master thesis](./assets/Master_Thesis.pdf).


## Contributions
This implementation adapts the Semantic Diffusion Model approach for semantic image synthesis using scarce real datasets. There are two major changes compared to the original implementation:

1. The tracking of training epochs and an improved model checkpointing based on the training loss of an epoch. (While it would be more common to use a separate validation set and thus the validation loss as a checkpointing criterion, this proved to be suboptimal for the small initial datasets that were tested.) This change also involves a maximal number of epochs after which training is aborted automatically (before it had to be stopped manually) and the avoidance to save (possible unusable and redundant) checkpoints every 10000 optimiser steps, both of which is beneficial for disk space.

2. The reduction of necessary sampling steps. Instead of using 1000 diffusion steps for sampling and training, it was found that the diffusion steps during sampling can be reduced to 300 (or less) if the "rescale_timesteps" parameter is set to true.

Instead of specifying the necessary scripts and parameters on the command line, all the possible actions (train, finetune, sample) are wrapped into the [main.py](./main.py) script and controlled via configuration files in JSON format. Find the installation and execution instructions below.


## Dependency Installation

Conda is used for dependency and virtual environment management. The code was tested on Linux and Windows.

### Linux
> conda env create -f linux_environment.yml

#### System Description
OS: Ubuntu 23.04

CPU: AMD Ryzen 7 5700X 8-Core Processor

RAM: 62 GB

GPU: NVIDIA GeForce RTX 3060 12 GB

### Windows
OS: Windows 10 Enterprise LTSC

CPU: Intel Core i9-10900X

RAM: 64 GB

GPU: NVIDIA GeForce RTX 3090 24 GB


## Execution

### Required Data Format
The SDMs need to be trained on real semantic segmentation datasets. So, it is required to have images and corresponding pixel-level labels (semantic masks). Such datasets come in different formats and the original code would also support for some of these, but for my master thesis I used a very simple format of saving both images and semantic masks as picture files. Images can be in PNG and JPG format and masks are required to be PNGs. They need to be saved in separate folders named "images" and "masks" and corresponding files need to share a filename. Furthermore, it is required to provide train and test datasets. The test dataset is only used during evaluation to check if the synthetic data actually helps to improve semantic segmentation models. Therefore, the final required folder structure should look like the following:

├── dataset_name
│   ├── test
│   │   ├── images
│   │   │   ├── **/*.jpg
│   │   │   ├── **/*.png
│   │   ├── masks
│   │   │   ├── **/*.png
│   ├── train
│   │   ├── images
│   │   │   ├── **/*.jpg
│   │   │   ├── **/*.png
│   │   ├── masks
│   │   │   ├── **/*.png

If you follow this structure then you can just set the configuration parameters `data_dir` to <dataset_name> and `dataset_mode` to "lemon-binary" or "lemon-multi-class" depending on if your dataset is a binary or multi-class segmentation dataset.

### Actions and Config Files
ISDM supports to perform three different actions ("train", "finetune" and "sample") all of which require slightly different configuration files.

1. Train: This action trains a SDM on the specified dataset. It is launched using the command:
> python main.py train -c configs/config_train.json

One will need to change the default [train configuration](./configs/config_train.json) according to his dataset. The parameters that (may) need to be changed are:
* data_dir: As explained above in section "Required Data Format"
* dataset_mode: As explained above in section "Required Data Format"
* batch_size: use smaller values to provide OOM exceptions
* image_size
* num_classes: set it to 2 if dataset_mode is "lemon-binary"
* resume_checkpoint: this is only used to resume a previous model training, don't specify this parameter for fresh training processes.

2. Finetune:



3. Sample:


## Showcase
asdf



## License
asdf


## Citation
asdf



-----original readme---- TO DELETE
# Semantic Image Synthesis via Diffusion Models (SDM)

&nbsp;

<img src='assets\results.png' align="left">  

&nbsp;

<img src='assets/diffusion.png' align="left">

&nbsp;

### [Paper](https://arxiv.org/abs/2207.00050)

[Weilun Wang](https://scholar.google.com/citations?hl=zh-CN&user=YfV4aCQAAAAJ), [Jianmin Bao](https://scholar.google.com/citations?hl=zh-CN&user=hjwvkYUAAAAJ), [Wengang Zhou](https://scholar.google.com/citations?hl=zh-CN&user=8s1JF8YAAAAJ), [Dongdong Chen](https://scholar.google.com/citations?hl=zh-CN&user=sYKpKqEAAAAJ), [Dong Chen](https://scholar.google.com/citations?hl=zh-CN&user=_fKSYOwAAAAJ), [Lu Yuan](https://scholar.google.com/citations?hl=zh-CN&user=k9TsUVsAAAAJ), [Houqiang Li](https://scholar.google.com/citations?hl=zh-CN&user=7sFMIKoAAAAJ),

## Abstract

We provide our PyTorch implementation of Semantic Image Synthesis via Diffusion Models (SDM). 
In this paper, we propose a novel framework based on DDPM for semantic image synthesis.
Unlike previous conditional diffusion model directly feeds the semantic layout and noisy image as input to a U-Net structure, which may not fully leverage the information in the input semantic mask,
our framework processes semantic layout and noisy image differently.
It feeds noisy image to the encoder of the U-Net structure while the semantic layout to the decoder by multi-layer spatially-adaptive normalization operators. 
To further improve the generation quality and semantic interpretability in semantic image synthesis, we introduce the classifier-free guidance sampling strategy, which acknowledge the scores of an unconditional model for sampling process.
Extensive experiments on three benchmark datasets demonstrate the effectiveness of our proposed method, achieving state-of-the-art performance in terms of fidelity (FID) and diversity (LPIPS).


## Example Results
* **Cityscapes:**

<p align='center'>  
  <img src='assets/cityscapes.png'/>
</p>

* **CelebA:**

<p align='center'>  
  <img src='assets/celeba.png'/>
</p>

* **ADE20K:**

<p align='center'>  
  <img src='assets/ade.png'/>
</p>

* **COCO-Stuff:**

<p align='center'>  
  <img src='assets/coco.png'/>
</p>

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Dataset Preparation
The Cityscapes and ADE20K dataset can be downloaded and prepared following [SPADE](https://github.com/NVlabs/SPADE.git). The CelebAMask-HQ can be downloaded from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), you need to to integrate the separated annotations into an image file (the format like other datasets, e.g. Cityscapes and ADE20K). 

### NEGCUT Training and Test

- Download the dataset.

- Train the SDM model:
```bash
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                   --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2  \
                                   --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
	                           --class_cond True --no_instance True
```

- Fine-tune the SDM model:
```bash
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-FINETUNE'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                   --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
                                   --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 --class_cond True \
                                   --no_instance True --drop_rate 0.2 --resume_checkpoint OUTPUT/ADE20K-SDM-256CH/model.pt
```

- Test the SDM model:
```bash
mpiexec -n 8 python image_sample.py --data_dir ./data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                    --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \ 
                                    --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 151 \
                                    --class_cond True --no_instance True --batch_size 2 --num_samples 2000 --s 1.5 \
                                    --model_path OUTPUT/ADE20K-SDM-256CH-FINETUNE/ema_0.9999_best.pt --results_path RESULTS/ADE20K-SDM-256CH
```

Please refer to the 'scripts/ade20.sh' for more details.

### Apply a pre-trained NEGCUT model and evaluate

#### Pretrained Models (to be updated)
|Dataset       |Download link     |
|:-------------|:-----------------|
|Cityscapes|[Visual results](https://drive.google.com/file/d/1TbLGCFJqRI4E8pFZJoHmj8MgDbwtjzhP/view?usp=sharing)|
|ADE20K|[Checkpoint](https://drive.google.com/file/d/1O8Avsvfc8rP9LIt5tkJxowMTpi1nYiik/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1NIXmrlBHqgyMHAoLBlmU8YELmL8Ij4kV/view?usp=sharing)|
|CelebAMask-HQ |[Checkpoint](https://drive.google.com/file/d/1iwpruJ5HMHdAA1tuNR8dHkcjGtxzSFV_/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1NDfU905iJINu4raoj4JdMOiHP8rTXr_M/view?usp=sharing)|
|COCO-Stuff |[Checkpoint](https://drive.google.com/file/d/17XhegAk8V5W3YiFpHMBUn0LED-n7B44Y/view?usp=sharing) \| [Visual results](https://drive.google.com/file/d/1ZluvN9spJF8jlXlSQ98ekWTmHrzwYCqo/view?usp=sharing)|

- To evaluate the model (e.g., ADE20K), first generate the test results:
```bash
mpiexec -n 8 python image_sample.py --data_dir ./data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 --diffusion_steps 1000 \
                                    --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 \ 
                                    --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 151 \
                                    --class_cond True --no_instance True --batch_size 2 --num_samples 2000 --s 1.5 \
                                    --model_path OUTPUT/ADE20K-SDM-256CH-FINETUNE/ema_0.9999_best.pt --results_path RESULTS/ADE20K-SDM-256CH
```

- To calucate FID metric, you should update "path1" and "path2" in "evaluations/test_with_FID.py" and run:
```bash
python evaluations/test_with_FID.py
```

- To calcuate LPIPS, you should evaluate the model for 10 times and run:
```bash
python evaluations/lpips.py GENERATED_IMAGES_DIR
```

### Acknowledge
Our code is developed based on [guided-diffusion](https://github.com/openai/guided-diffusion). We also thank "test_with_FID.py" in [OASIS](https://github.com/boschresearch/OASIS) for FID computation, "lpips.py" in [stargan-v2](https://github.com/clovaai/stargan-v2) for LPIPS computation.
