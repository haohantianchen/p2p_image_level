import argparse
import os

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.pipeline
import p2p.ptp_utils
import p2p.seq_aligner

parser = argparse.ArgumentParser()
parser.add_argument('--inverse', default=1, type=int ,help='whether perform null-text inversion')
parser.add_argument('--edit', default=0, type=int, help='whether perform p2p edit')
args = parser.parse_args()

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

LOW_RESOURCE = False
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
# MODEL = "runwayml/stable-diffusion-v1-5"
MODEL = "/home/jianshu/code/prompt_travel/data/models/sd/cyberrealistic_v33.safetensors"
height = 512
width = 512
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
sample_steps = 50
os.makedirs("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4", exist_ok=True)

# pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_pretrained(MODEL, scheduler=scheduler).to(device)
pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_single_file(MODEL,use_safetensors=True).to(device)
pipe.scheduler = scheduler
pipe.disable_xformers_memory_efficient_attention()
pipe.safety_checker = None
null_inversion = p2p.nti.NullInversion(pipe, num_ddim_steps=sample_steps)

# real image path and its prompt
image_path = "/raid/cvg_data/lurenjie/P2P/prompt_travel/stylize/warp_test/output_classify_body_a085/50_512.png"
# image_path = "/home/jianshu/code/prompt_travel/stylize/warp_test/00000000.png"
prompt = "a girl in beige top, black skirt, black boots"
if args.inverse:
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), num_inner_steps=100, verbose=True, resize=False)
    torch.save(x_t, "/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/xt_girl_512_512_50.pt")
    torch.save(uncond_embeddings, "/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/unc_girl_512_512_50.pt")
x_t, uncond_embeddings = torch.load("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/xt_girl_512_512_50.pt"), torch.load("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/unc_girl_512_512_50.pt")

x_t = x_t.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)

print("null-text inversion complete!")

prompts = [prompt]
# original prompt and new prompt
prompts = ["a girl in beige top, black skirt, black boots",
           "a girl in beige top, black skirt, black boots"
        ]
store_controller = p2p.p2p.AttentionStore(pipe.tokenizer)
if args.edit:
    p2p.ptp_utils.register_attention_control(pipe, store_controller)
images, all_step_latents = pipe(prompt=prompts[0], latents=x_t, height=height, width=width, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings, num_inverse_steps=sample_steps)
images[0].save("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/rec_girl_50steps_inverse50.png")

all_step_imgs = []
with torch.no_grad():
    for x in all_step_latents[::2]:
        all_step_imgs.append(pipe.vae.decode(x / pipe.vae.config.scaling_factor, return_dict=False)[0])
all_step_imgs = torch.cat(all_step_imgs)
do_denormalize = [True] * all_step_imgs.shape[0]
all_step_imgs = pipe.image_processor.postprocess(all_step_imgs, output_type="pil", do_denormalize=do_denormalize)
all_step_imgs = p2p.ptp_utils.merge_images(all_step_imgs)
all_step_imgs.save("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/rec_girl_all_steps.png")

# replace_controller = p2p.pipeline.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=device)
# p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings).images
# images[0].save("p2p_vis/rec_cat2.png")
# p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)

if args.edit:
    # blend_word = ((('cat',), ("dog",)))
    # eq_params = {"words": ("tiger",), "values": (2,)}
    # lb = p2p.p2p.LocalBlend(pipe.tokenizer, sample_steps, prompts, blend_word)
    lb = None
    replace_controller = p2p.p2p.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=lb, device=device)

    p2p.ptp_utils.register_attention_control(pipe, replace_controller)
    images, _ = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings, controller=replace_controller, ref_latents=all_step_latents)
    images[0].save("/raid/cvg_data/lurenjie/P2P/prompt_travel/xt_unc_4/rec_boy.png")
    p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)
    replace_controller.attention_store = replace_controller.ref_attention_store
    p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)


