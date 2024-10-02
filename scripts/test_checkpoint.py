import os
import torch
import numpy as np

from omegaconf import OmegaConf
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from unipaint.pipelines.pipeline_unipaint import AnimationPipeline
from unipaint.models.unet import UNet3DConditionModel
from unipaint.models.unipaint.brushnet import BrushNetModel


from unipaint.utils.util import load_weights,save_videos_grid
from unipaint.utils.mask import StaticRectangularMaskGenerator, MovingRectangularMaskGenerator, MarginalMaskGenerator, InterpolationMaskGenerator
from unipaint.utils.convert_to_moe import task_context, replace_ffn_with_moeffn

import decord
decord.bridge.set_bridge("torch")

def load_models(path, brushnet_path, motion_module_path, adapter_path,unet_checkpoint_path, device, dtype):
    #load base model
    tokenizer        = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder     = CLIPTextModel.from_pretrained(path, subfolder="text_encoder").to(device,dtype)
    vae              = AutoencoderKL.from_pretrained(path, subfolder="vae").to(device, dtype)

    inference_config = OmegaConf.load("configs/inference/inference-v3.yaml")
    unet             = UNet3DConditionModel.from_pretrained_2d(path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device, dtype)

    #load brushnet
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=dtype).to(device)

    #build pipeline
    pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                brushnet = brushnet,
                scheduler=DDIMScheduler(beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule="linear",
                                        steps_offset=0,
                                        clip_sample=False)
                                        ).to(device)

    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = motion_module_path,
        motion_module_lora_configs = [],
        # domain adapter
        adapter_lora_path          = adapter_path,
        adapter_lora_scale         = 1.0,
        # image layers
        dreambooth_model_path      = "",
        lora_model_path            = "",
        lora_alpha                 = 0.8,
    ).to(device)

    if use_moe:
        unet = replace_ffn_with_moeffn(unet)

    if (unet_checkpoint_path is not None) and (len(unet_checkpoint_path) > 0):
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')  # Remove 'module.' from each key
            new_state_dict[new_key] = v
        del(state_dict)
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        del(new_state_dict)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    brushnet.requires_grad_(False)
    return pipeline

def do_task(pipeline, task, prompt, n_prompt, video_path, mask_data, save_path, device, dtype):
    vr = decord.VideoReader(video_path, width=512, height=512)
    video = vr.get_batch(list(range(0,16)))
    video = rearrange(video, "f h w c -> c f h w")
    frame = torch.clone(torch.unsqueeze(video/255, dim=0)).to(device, dtype)
    del(vr)

    if task == "static_inpaint":
        mask_generator = StaticRectangularMaskGenerator(mask_l=(0.1,0.4),
                                                        mask_r=(0.1,0.4),
                                                        mask_t=(0.1,0.4),
                                                        mask_b=(0.1,0.4),)
        mask = mask_generator(frame)
        context = "inpaint"
    elif task == "moving_inpaint":
        mask_generator = MovingRectangularMaskGenerator(rect_height_range=(0.2,0.5),
                                                        rect_width_range=(0.2,0.5))
        mask = mask_generator(frame)
        context = "inpaint"
    elif task == "outpaint":
        mask_generator = MarginalMaskGenerator(mask_l=(0.0,0.3),
                                            mask_r=(0.0,0.3),
                                            mask_t=(0.0,0.3),
                                            mask_b=(0.0,0.3),)
        mask = mask_generator(frame)
        context = "outpaint"
    elif task == "interpolation":
        mask_generator = InterpolationMaskGenerator(stride_range=(2,2))
        mask = mask_generator(frame)
        context = "interpolation"
    elif task == "segment_inpaint":
        mask = torch.tensor(np.unpackbits(mask_data['mask']).reshape((1,1,16, 512, 512))).expand(1,3,16,512,512)
        context = "inpaint"
    elif task == "segment_outpaint":
        mask = torch.tensor(np.unpackbits(mask_data['mask']).reshape((1,1,16, 512, 512))).expand(1,3,16,512,512)
        mask = 1 - mask
        context = "outpaint"

    frame[mask==1]=0
    mask = mask.to(device, dtype)
    frame = frame*2.-1.
    mask = mask*2.-1.
    with torch.no_grad():
        with task_context(context):
            sample = pipeline(
                prompt = prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = 25,
                guidance_scale      = 12.5,
                width               = 512,
                height              = 512,
                video_length        = 16,

                init_video = frame[:,:,:],
                mask_video = mask[:,:,:],
                brushnet_conditioning_scale = 1.0,
                control_guidance_start = 0.0,
                control_guidance_end = 1.0,
                ).videos
    sample = torch.concat([((frame+1)/2).cpu(), sample])
    save_videos_grid(sample, save_path)

#running config
device = "cuda:3"
dtype = torch.float16

#model config
use_motion_module = False
use_adapter = False
use_moe = True

path = "models/StableDiffusion/stable-diffusion-v1-5"
brushnet_path = "models/BrushNet/random_mask_brushnet_ckpt"
motion_module_path = "models/Motion_Module/v3_sd15_mm.ckpt" if use_motion_module else ""
adapter_path = "models/Motion_Module/v3_sd15_adapter.ckpt" if use_adapter else ""
unet_checkpoint_path = "checkpoints/mixed_4.ckpt"

pipeline = load_models(path, brushnet_path, motion_module_path, adapter_path,unet_checkpoint_path,device, dtype)

#data config
data_folder = "CI"
vid = "Women6"
scene_prompt = "a church hall with white greece columns"
segment_prompt = ""

video_path = f"outpaint_videos/{data_folder}/{data_folder}_{vid}.mp4"
mask_data = np.load(f'outpaint_videos/{data_folder}/{data_folder}_{vid}.npz')
addtional_prompt = "Extremely realistic, high resolution."
n_prompt = "worst quality, low quality, letterboxed, wood sticks, random shade"

tasks = ["static_inpaint", "moving_inpaint", "segment_inpaint", "outpaint", "interpolation"]
tasks = ["segment_outpaint"]
for task in tasks:
    save_path = f"./samples/{vid}/{task}.gif"
    if task == "segment_inpaint":
        prompt = segment_prompt + addtional_prompt
    else:
        prompt = scene_prompt + addtional_prompt
    do_task(pipeline, task, prompt, n_prompt, video_path, mask_data, save_path, device, dtype)
    
