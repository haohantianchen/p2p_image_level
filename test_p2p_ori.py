import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.p2p
import p2p.ptp_utils
import p2p.seq_aligner

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
MODEL = "runwayml/stable-diffusion-v1-5"
controlnet_path = "/home/jianshu/code/sd/controlnet/densepose/checkpoint-17000/controlnet"
img_path = "/home/jianshu/code/video_sd_controlnet_densepose/densepose_scripts/detectron2/projects/DensePose/output_js/gen.png"
controlnet = ControlNetModel.from_pretrained(controlnet_path, load_files_only=True)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     MODEL, controlnet=controlnet,
# ).to(device)
pipe = p2p.nti.ReconstructStableDiffusionPipeline.from_pretrained(MODEL).to(device)

# ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
tokenizer = pipe.tokenizer
height, width = 512, 512
sample_steps = 50

g_cpu = torch.Generator().manual_seed(8888)
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]

controller = p2p.p2p.AttentionReplace(tokenizer, prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=1)
controlnet_controller = p2p.p2p.AttentionStore(tokenizer)
p2p.ptp_utils.register_attention_control(pipe, controller)

pipe.safety_checker = None
latent = torch.randn(
    (1, pipe.unet.in_channels, height // 8, width // 8),
    generator=g_cpu,
    # dtype=torch.float16
)
latents = latent.expand(len(prompts), pipe.unet.in_channels, height // 8, width // 8).to(device)
images = pipe(prompt=prompts, latents=latents, num_inference_steps=sample_steps).images
images[0].save("0.png")

p2p.p2p.show_cross_attention(prompts, controller, res=16, from_where=("up", "down"), select=1, )

