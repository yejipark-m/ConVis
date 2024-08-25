import json
import os
from PIL import Image
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from compel import Compel, ReturnedEmbeddingsType

import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="minigpt4")
parser.add_argument("-m", type=int, default=256)
parser.add_argument("--mme_type", type=str, default="artwork")
parser.add_argument(
    "--benchmark_name", type=str, default="CHAIR")
parser.add_argument(
    "--caption_path", type=str, help="Path of the caption file that you want to generate image with.")

parser.add_argument(
    "--output_path", type=str, default='./generated_images', help="Path of the output.")

args = parser.parse_known_args()[0]
model_type = args.model_type
mme_type = args.mme_type
benchmark_name = args.benchmark_name
caption_path = args.caption_path
output_path = args.output_path

if benchmark_name == "MME":
    output_dir = f"{output_path}/{benchmark_name}/{mme_type}/{model_type}/num_steps_1"
elif benchmark_name == "demo":
    output_dir = output_path
else:
    output_dir = f"{output_path}/{benchmark_name}/{model_type}/num_steps_1"


base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"

unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device="cuda"))
pipeline = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
if args.m > 77:
    compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

os.makedirs(output_dir, exist_ok=True)

with open(caption_path, "r") as file:
    lines = file.readlines()

total_time = 0.0

for i, line in enumerate(lines):
    data = json.loads(line)
    caption = data["caption"]
    if args.m > 77:
        conditioning, pooled = compel(caption)

    start_time = time.time()
    if args.m > 77:
        image = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, num_inference_steps=1, guidance_scale=0, timesteps=[800]).images[0]
    else:
        image = pipeline(prompt=caption, num_inference_steps=1, guidance_scale=0, timesteps=[800]).images[0]


    end_time = time.time()
    total_time += end_time - start_time


    def get_unique_filename(filepath):
        base, ext = os.path.splitext(filepath)
        counter = 1
        new_filepath = filepath
        while os.path.exists(new_filepath):
            new_filepath = f"{base}_{counter}{ext}"
            counter += 1
        return new_filepath


    if benchmark_name in ('CHAIR', 'POPE', 'MME'):
        image_name = f"image_{data['image_id']}.png"
    elif benchmark_name == 'HallusionBench':
        image_name = data["image_path"]
    elif benchmark_name == 'LLaVA-Bench':
        image_name = f"image_{data['answer_id']}.png"

    image_name = image_name.replace(".png", "_t2i.png")
    image_path = os.path.join(output_dir, image_name)

    # Use the get_unique_filename function to generate a unique file name
    unique_image_path = get_unique_filename(image_path)
    image.save(unique_image_path)


print(f"Images generation time: {total_time:.2f} seconds")

