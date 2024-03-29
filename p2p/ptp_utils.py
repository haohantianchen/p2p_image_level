# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from tqdm.notebook import tqdm

"""
输入：list(PIL.Image)
"""
def merge_images(image_list, rows=1, cols=-1):
    # 检查输入的图像数量是否符合行列要求
    if cols==-1 and len(image_list)%rows == 0:
        cols = len(image_list) // rows
    assert len(image_list) == rows * cols, "输入图像数量与指定的行列不匹配"

    # 获取图像的宽度和高度
    img_width, img_height = image_list[0].size

    # 创建新的图像
    merged_image = Image.new("RGB", (cols * img_width, rows * img_height))

    # 按行列合并图像
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            merged_image.paste(image_list[index], (col * img_width, row * img_height))

    return merged_image


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save_dir="./p2p_vis", save_name="test.png"):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(f"{save_dir}/{save_name}"):
        for i in range(10000):
            save_name = "_".join(save_name.split(".")[0].split("_")[:-1]) + f"_{i}.png"
            if not os.path.exists(f"{save_dir}/{save_name}"):
                break
    pil_img.save(f"{save_dir}/{save_name}")
    print(f"successfully save image in:{save_dir}/{save_name}")
    return image_
    #display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller, is_controlnet=False):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            if attention_mask is not None:
                print("attention_mask:", attention_mask.shape)
            residual = hidden_states

            with torch.no_grad():
                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                attention_probs = controller(attention_probs, is_cross, place_in_unet)

                # value = value.to(torch.float16)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = to_out(hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states

        @torch.no_grad()
        def forward_(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            if attention_mask is not None:
                print("attention_mask:", attention_mask.shape)
            residual = hidden_states

            with torch.no_grad():
                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                frames = len(controller)
                p2p_batchsize = hidden_states.shape[0] // frames
                assert p2p_batchsize >= 1

                encoder_hidden_states = list(torch.chunk(encoder_hidden_states, p2p_batchsize))
                for i in range(p2p_batchsize):
                    encoder_hidden_states[i] = torch.chunk(encoder_hidden_states[i], frames)
                hidden_states = list(torch.chunk(hidden_states, p2p_batchsize))
                for i in range(p2p_batchsize):
                    hidden_states[i] = torch.chunk(hidden_states[i], frames)
                if attention_mask is not None:
                    attention_mask = list(torch.chunk(attention_mask, p2p_batchsize))
                    for i in range(p2p_batchsize):
                        attention_mask[i] = torch.chunk(attention_mask[i], frames)
                final_hidden_states = []
                for i in range(frames):
                    hidden_states_ = torch.cat([x[i] for x in hidden_states])
                    encoder_hidden_states_ = torch.cat([x[i] for x in encoder_hidden_states])
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([x[i] for x in attention_mask])
                    else:
                        attention_mask_ = None
                    query = self.to_q(hidden_states_)
                    key = self.to_k(encoder_hidden_states_)
                    value = self.to_v(encoder_hidden_states_)

                    query = self.head_to_batch_dim(query)
                    key = self.head_to_batch_dim(key)
                    value = self.head_to_batch_dim(value)

                    attention_probs = self.get_attention_scores(query, key, attention_mask_)
                    attention_probs = controller[i](attention_probs, is_cross, place_in_unet)

                    hidden_states_ = torch.bmm(attention_probs, value)
                    hidden_states_ = self.batch_to_head_dim(hidden_states_)
                    final_hidden_states.append(hidden_states_)

                tmp = []
                for i in range(frames):
                    tmp.append(torch.chunk(final_hidden_states[i], p2p_batchsize))
                final_hidden_states = []
                for i in range(p2p_batchsize):
                    final_hidden_states.append(torch.cat([x[i] for x in tmp]))
                hidden_states = torch.cat(final_hidden_states)

                # linear proj
                hidden_states = to_out(hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
        if isinstance(controller, list):
            return forward_
        else:
            return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    if is_controlnet:
        sub_nets = model.named_children()
    else:
        sub_nets = model.unet.named_children()
    if is_controlnet:
        sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    if isinstance(controller, list):
        for i in range(len(controller)):
            controller[i].num_att_layers = cross_att_count
    else:
        controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def prepare_image(img_path, width=512, height=512):
    image = Image.open(img_path).convert("RGB")
    width, height = (
            x - x % 8 for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor
    image = image.resize((width, height))
    images = [image]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)
    """
    Convert a numpy image to a pytorch tensor
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

@torch.no_grad()
def controlnet_diffusion_step(model, controller, controlnet_controller, image, latents, context, t, guidance_scale, low_resource=False):
    latents_input = torch.cat([latents] * 2)
    down_block_res_samples, mid_block_res_sample = model.controlnet(
        latents_input,
        t,
        encoder_hidden_states=context,
        controlnet_cond=image,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict = False
    )
    noise_pred = model.unet(
        latents_input,
        t,
        encoder_hidden_states=context,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

@torch.no_grad()
def controlnet_stablediffusion(
    model,
    img_path,
    prompt: List[str],
    controller,
    controlnet_controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    register_attention_control(model.controlnet, controlnet_controller, True)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    image = prepare_image(img_path).to(model.device)
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = controlnet_diffusion_step(
            model,
            controller,
            controlnet_controller,
            image,
            latents,
            context,
            t,
            guidance_scale,
            low_resource
        )

    image = latent2image(model.vae, latents)

    return image, latent

# def checkpoint_to_pipeline(
#     checkpoint: Path,
#     target_dir: Optional[Path] = None,
#     save: bool = True,
# ) -> StableDiffusionPipeline:
#     # logger.debug(f"Converting checkpoint {path_from_cwd(checkpoint)}")
#     if target_dir is None:
#         target_dir = pipeline_dir.joinpath(checkpoint.stem)

#     pipeline = StableDiffusionPipeline.from_single_file(
#         pretrained_model_link_or_path=str(checkpoint.absolute()),
#         local_files_only=True,
#         load_safety_checker=False,
#     )

#     if save:
#         target_dir.mkdir(parents=True, exist_ok=True)
#         # logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
#         pipeline.save_pretrained(target_dir, safe_serialization=True)
#     return pipeline, target_dir

# def get_checkpoint_weights(checkpoint: Path):
#     temp_pipeline: StableDiffusionPipeline
#     temp_pipeline, _ = checkpoint_to_pipeline(checkpoint, save=False)
#     unet_state_dict = temp_pipeline.unet.state_dict()
#     tenc_state_dict = temp_pipeline.text_encoder.state_dict()
#     vae_state_dict = temp_pipeline.vae.state_dict()
#     return unet_state_dict, tenc_state_dict, vae_state_dict