import os
import torch
from omegaconf import OmegaConf

from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from unipaint.pipelines.pipeline_unipaint import AnimationPipeline
from unipaint.models.unet import UNet3DConditionModel
from unipaint.models.unipaint.brushnet import BrushNetModel
import numpy as np

from unipaint.utils.util import load_weights,save_videos_grid
import decord
decord.bridge.set_bridge("torch")
from unipaint.utils.mask import StaticRectangularMaskGenerator

path = "models/StableDiffusion/stable-diffusion-v1-5"
brushnet_path = "models/BrushNet/random_mask_brushnet_ckpt"
device = "cuda:2"
dtype = torch.float16

use_motion_module = True
use_adapter = True

motion_module_path = "models/Motion_Module/v3_sd15_mm.ckpt" if use_motion_module else ""
adapter_path = "models/Motion_Module/v3_sd15_adapter.ckpt" if use_adapter else ""

data = np.load('SB_Eagle_mask.npz')
save_path = "./samples/"

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

unet_checkpoint_path = "outputs/unipaint_training-2024-08-14T10-26-32/checkpoints/checkpoint-epoch-5.ckpt"
unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
if "global_step" in unet_checkpoint_path: print(f"global_step: {unet_checkpoint_path['global_step']}")
state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  # Remove 'module.' from each key
    new_state_dict[new_key] = v

m, u = unet.load_state_dict(new_state_dict, strict=False)

unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
brushnet.requires_grad_(False)

video_path = "outpaint_videos/SB_Eagle.mp4"
vr = decord.VideoReader(video_path, width=512, height=512)

video = vr.get_batch(list(range(0,16)))
video = rearrange(video, "f h w c -> c f h w")
frame = torch.clone(torch.unsqueeze(video/255, dim=0)).to(device, brushnet.dtype)
del(vr)


mask = torch.tensor(np.unpackbits(data['mask']).reshape((16, 512, 512)))
mask = torch.cat([mask.unsqueeze(dim=0)]*3,dim=0).unsqueeze(dim=0)
frame[mask==1]=0
mask = mask.to(device, brushnet.dtype)
frame = frame*2.-1.
mask = mask*2.-1.
save_videos_grid(((frame+1)/2).cpu(), save_path+"test_masked_video_eagle.gif")
prompt = "a white head Bald Eagle"
n_prompt = ""
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
save_videos_grid(sample, save_path+"sam_test_eagle.gif")
