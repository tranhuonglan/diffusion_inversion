# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import numpy as np

from diffusers.models.modeling_utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict, ConfigMixin, register_to_config
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput
import torchvision.transforms as transforms
# from torchvision import datasets
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from models.resnet import ResNet18

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import argparse
import os
import ml_collections
from datasets import load_dataset
from dataset import *
from classes import i2d
import PIL

PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

def syn_images(num_images_per_class=500,
                num_classes=10, 
               prompts=[],
               num_images_per_prompt=10,
               num_inference_steps=50,
               pretrain_model_name='CompVis/stable-diffusion-v1-4',
               root_name='./results/tiny-imagenet-syn-baseline-group2',
               labels=None,):
    print(">>>> Number of Image per Class, number of images per prompt", num_images_per_class, num_images_per_prompt)
    pipe =  StableDiffusionPipeline.from_pretrained(pretrain_model_name, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    for i in range(num_classes):
        id = 0
        print(f'{prompts[i]}')
        prompt = prompts[i]
        for _ in range((num_images_per_class//num_images_per_prompt) + (num_images_per_class%num_images_per_prompt!=0)):
            images = pipe(prompt, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt).images
            
            for img in images:
                # img = img.resize((save_resolution, save_resolution),
                #                 resample=PIL_INTERPOLATION[interpolation])
                try:
                    if not os.path.exists(f'{root_name}/class_{labels[i]:04d}'):
                        os.makedirs(
                                f'{root_name}/class_{labels[i]:04d}', exist_ok=True)
                    img.save(
                        f'{root_name}/class_{labels[i]:04d}/{id:04d}.png')
                    id+=1
                except:
                    raise Exception('Error in saving image')

def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def main():
    parser = argparse.ArgumentParser(description="Script Description")
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators.')
    parser.add_argument('--pretrain_model_name', type=str, default='CompVis/stable-diffusion-v1-4', help='Pretrained model name.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps.')
    parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images generated per prompt.')
    parser.add_argument('--dataset_name', type=str, default='tiny-imagenet', help='Name of the dataset.')
    parser.add_argument('--data_dir', type=str, default='~/tensor_flow', help='Directory for the dataset.')
    parser.add_argument('--size', type=int, default=32, help='Image size.')
    parser.add_argument('--num_images_per_class', type=int, default=1000, help='Size of the training set.')
    parser.add_argument('--root_name', type=str, default='./results/tiny-imagenet-syn-baseline-group2', help='Root name for saving results.')
    parser.add_argument('--first_half', action='store_true', help='Process the first half of the data.')

    args = parser.parse_args()
    
    # Use args to set the environment and other variables
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    pretrain_model_name = args.pretrain_model_name
    num_inference_steps = args.num_inference_steps
    num_images_per_prompt = args.num_images_per_prompt
    dataset_name = args.dataset_name
    data_dir = args.data_dir
    root_name = args.root_name
    first_half = args.first_half
    num_images_per_class = args.num_images_per_class
    
    # Load the dataset
    if dataset_name == 'tiny-imagenet':
        from split_class import subset_tiny_img
        #tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')
        ds = load_dataset('Maysee/tiny-imagenet', split='valid')
        group1 = subset_tiny_img
        class_names = [i2d[id] for id  in ds.features['label'].int2str(group1)]
        print(group1)
        print(class_names)
        
        num_classes = len(group1)
        if first_half:
            print(group1[:num_classes//2])
            print('>>>> Generate the first half of the dataset <<<<', class_names[:num_classes//2])
            syn_images(num_images_per_class=num_images_per_class,prompts=class_names[:num_classes//2],
                       num_classes=num_classes//2, 
                       labels=group1[:num_classes//2],
                       num_images_per_prompt=num_images_per_prompt, 
                       num_inference_steps=num_inference_steps, 
                       pretrain_model_name=pretrain_model_name, 
                       root_name=root_name)
                       #pretrained_classifier=pretrained_dir)
        else:
            print(group1[num_classes//2:])
            print('>>>> Generate the second half of the dataset <<<<', class_names[num_classes//2:])
            syn_images(num_images_per_class=num_images_per_class,prompts=class_names[num_classes//2:], num_classes=num_classes//2,
                       num_images_per_prompt=num_images_per_prompt, 
                       labels=group1[num_classes//2:],
                       num_inference_steps=num_inference_steps, 
                       pretrain_model_name=pretrain_model_name, 
                       root_name=root_name)
                       #pretrained_classifier=pretrained_dir)

    if dataset_name == 'cifar10':
        config = D(
                name=dataset_name,
                data_dir='~/tensorflow_datasets',
            )
        _, y_train = get_dataset(config, return_raw=True, train_only=True)
        print(config.class_names)

if __name__ == "__main__":

    main()
   
    

