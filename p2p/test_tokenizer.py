import torch
from diffusers import StableDiffusionPipeline

# import p2p
# import p2p_utils

MODEL = "runwayml/stable-diffusion-v1-5" 
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
tokenizer = ldm_stable.tokenizer

prompts = ["A painting of a squirrel eating a burger", "A painting of a tiger eating a burger"]
tokens = tokenizer.encode(prompts[0])
print("tokens:")
print(tokens)

print("decoded tokens:")
print(tokenizer.decode(tokens))

text_inputs = tokenizer(
    prompts[0],
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
print(text_inputs)
text_input_ids = text_inputs.input_ids
print(text_input_ids.shape)
print(text_input_ids)
prompt_embeds = ldm_stable.text_encoder(text_input_ids.to(device))[0]
print(prompt_embeds.shape)
print(prompt_embeds)
uncond_input = tokenizer([""] * 1, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
print(uncond_input.input_ids)
uncond_embeds = ldm_stable.text_encoder(uncond_input.input_ids.to(device))[0]
print(uncond_embeds)
print(uncond_embeds.shape)

words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(prompts[0])][1:-1]
print(words_encode)