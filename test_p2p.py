import os
import time

import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.p2p
import p2p.pipeline
import p2p.ptp_utils
import p2p.seq_aligner

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODEL = "runwayml/stable-diffusion-v1-5"
MODEL = "/home/jianshu/code/prompt_travel/data/models/sd/cyberrealistic_v33.safetensors"
dtype = torch.float16


# pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype).to(device)
pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_single_file(MODEL, torch_dtype=dtype).to(device)
tokenizer = pipe.tokenizer
height, width = 768, 1024
sample_steps = 50

g_cpu = torch.Generator().manual_seed(8888)
# prompts = ["A painting of a squirrel eating a burger",
#            "A painting of a squirrel eating a pizza",
#            ] + ["A painting of a squirrel eating a pizza" ]*0
# prompts = [
#     " a photo of a living room, 4k, HD",
#     # " solo, 1girl, beautiful girl, shorts, white tank-top,, smiling, ((best quality)), ((masterpiece)), ((realistic)), "
#     " a photo of a girl in white tank-top, shorts"
# ]
prompts = ["a girl in beige top, black skirt, black boots, stands still with arms hanging straight on her sides",
           "a girl in beige top, black skirt, black boots"
        ]


pipe.safety_checker = None
latent = torch.randn(
    (1, pipe.unet.in_channels, height // 8, width // 8),
    generator=g_cpu,
    dtype=dtype
)
latents = latent.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)
images, ref_latents = pipe(prompt=prompts[0], latents=latents, num_inference_steps=sample_steps)

show_img = [images[0]]

store_controller = p2p.p2p.AttentionStore(tokenizer, device)

p2p.ptp_utils.register_attention_control(pipe, store_controller)
images, _ = pipe(prompt=prompts[1], latents=latents, num_inference_steps=sample_steps)
images[0].save("/raid/cvg_data/lurenjie/P2P/p2p_vis/girl.png")
show_img.append(images[0])

p2p.p2p.show_cross_attention(prompts, store_controller, res=16, from_where=["up", "down"], select=1, save_dir="/raid/cvg_data/lurenjie/P2P/p2p_vis")

latents = latent.expand(len(prompts)-1, pipe.unet.in_channels, height // 8, width // 8).to(device)
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_background_steps = [0,sample_steps]
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)
# del store_controller
# torch.cuda.empty_cache()
# replace_controller = []
# for i in range(1):
#     replace_controller.append(p2p.p2p.AttentionReplace(store_controller.all_attn_store, tokenizer,prompts[1:]*2, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=device))
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:], latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    ref_latents=ref_latents,
    replace_background_steps=replace_background_steps
)
# images[0].save("p2p_vis/shift_pizza.png")
# images[0].save("p2p_vis/2_pizza1.png")
show_img.append(images[0])

replace_background_steps = [0,sample_steps-5]
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:], latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    ref_latents=ref_latents,
    replace_background_steps=replace_background_steps
)
show_img.append(images[0])

replace_background_steps = [0,sample_steps-10]
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:], latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    ref_latents=ref_latents,
    replace_background_steps=replace_background_steps
)
show_img.append(images[0])

replace_background_steps = [0,sample_steps-10]
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:], latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    ref_latents=ref_latents,
    replace_background_steps=replace_background_steps
)
show_img.append(images[0])

replace_background_steps = [-1, -1]
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:], latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    ref_latents=ref_latents,
    replace_background_steps=replace_background_steps
)
show_img.append(images[0])
img = p2p.ptp_utils.merge_images(show_img)
img.save("p2p_vis/show_replace_bg_50.png")
# images[1].save("p2p_vis/2_pancake1.png")



# p2p.p2p.show_cross_attention(prompts, store_controller, res=16, from_where=("up", "down"), select=1, save_dir="./1109")
# p2p.p2p.show_cross_attention(prompts, controlnet_controller, res=16, from_where=["down"], select=1, save_dir="./1109")
