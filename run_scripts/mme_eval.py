import argparse
import os
import random
import sys
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from PIL import Image
import json
import glob

from types import SimpleNamespace

from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria



MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}


def setup_seeds(image_seed, model_seed):
    random.seed(image_seed)
    np.random.seed(image_seed)
    torch.manual_seed(model_seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="greedy",
    help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
)

parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-oxoptions instead.",
)

parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="data path",
)
parser.add_argument(
    "--images_path",
    type=str,
    default="generated_images/",
    help="generated_images_path"
)
parser.add_argument(
    "--diff_num_step",
    type=int,
    default=1,
    help="diffusion num_steps for image generation"
)

parser.add_argument("-b", "--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--image_seed", type=int, default=42)
parser.add_argument("--model_seed", type=int, default=42)
parser.add_argument("-m", "--max_new_tokens", type=int, default=128)

parser.add_argument(
    "-v",
    "--verbosity",
    action="store_false",
    dest="verbosity",
    default=True,
    help="Verbosity. Default: True.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./generated_captions/",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)
parser.add_argument(
    "--mme_type",
    type=str,
    default='existence',
)
parser.add_argument(
    "-p",
    "--post-correction",
    type=str,
    default=None,
    help="Post correction method such as woodpecker, lure.",
)
parser.add_argument(
    "-e",
    "--expand-ratio",
    type=float,
    default=0.6,
    help="Expand ratio of growing contextual field.",
)

### ConVis Argument Starts ###
parser.add_argument(
    "--convis_alpha", type=float, default=0.1, help="Weight for contrast"
)
parser.add_argument("--num_of_images", type=int, default=4, help="Number of used generated images.")
parser.add_argument(
    "--cutoff_lambda", type=float, default=None
)
### ConVis Argument Ends ###

parser.add_argument("--skip_num", type=int, default=0, help="Skip the first skip_num samples.")
parser.add_argument(
    "--debugger",
    action="store_true",
    default=False,
    help="Whether to use debugger output.",
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="To record results.json with exp_id=current_time",
)

args = parser.parse_known_args()[0]

args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)

model_name = args.model
decoding_strategy = args.decoder

setup_seeds(args.image_seed, args.model_seed)

device = 'cuda'

verbosity = args.verbosity
sample = args.sample
top_p = args.top_p
temperature = args.temperature
data_path = args.data_path
mme_type = args.mme_type
output_dir = args.output_dir
num_beams = args.beam
max_new_tokens = args.max_new_tokens
debugger = args.debugger
skip_num = args.skip_num
convis_alpha = args.convis_alpha
cutoff_lambda = args.cutoff_lambda
num_of_images = args.num_of_images
exp_id = args.exp_id

# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

model_config = cfg.model_cfg
device_map = {'device_map': 'auto'}
model_config.device_8bit = device_map
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)
if model_name in {"minigpt4", "instructblip"}:
    model = model.to(device)
model.eval()
print("model device", model.device)

processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)

if model_name == 'mplug-owl2':
    vis_processors["eval"] = model.image_processor

valid_decoding_strategies = [
        "greedy",
        "beam",
        "sample",
        "convis",
    ]


valid_post_editing_strategies = ["lure", "woodpecker"]
valid_detector = ["dino", "owlv2"]

assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"

decoding_strategy = decoding_strategy
convis_decoding = False
beam_search = False

print("decoding_strategy", decoding_strategy)
if decoding_strategy == "greedy":
    pass
elif decoding_strategy == "beam":
    beam_search = True
elif decoding_strategy == 'sample':
    sample = True
elif decoding_strategy == "convis":
    convis_decoding = True

print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")


if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("data_path: ", data_path)
    print("output_dir: ", output_dir)
    print("num_beams: ", num_beams)
    print("seed: ", args.image_seed)


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

data_path = os.path.join(data_path, mme_type)
question_path = data_path + f"/{mme_type}_questions.jsonl"

generated_image_file_path = os.path.join(args.images_path, mme_type, model_name, f"num_steps_{args.diff_num_step}")
print(generated_image_file_path)

base_dir = os.path.join(output_dir, "mme", model_name, decoding_strategy)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


with open(question_path, 'r', encoding='utf-8') as question_list:
    lines = question_list.readlines()  # Read all lines first to ensure tqdm works
    for idx, line in enumerate(tqdm(lines, unit="line")):
        data = json.loads(line.strip())
        img_file = data["image"]
        qu_raw = data["text"]
        label = data["label"]

        image_path = os.path.join(data_path, img_file)
        raw_image = Image.open(image_path).convert('RGB')

        if model_name == "mplug-owl2":
            max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
            image = raw_image.resize((max_edge, max_edge))
            image_tensor = process_images([image], model.image_processor)
            image = image_tensor.to(device, dtype=torch.float16)
        else:
            image = vis_processors["eval"](raw_image).unsqueeze(0)
            image = image.to(device)

        if convis_decoding:
            pattern = os.path.join(generated_image_file_path, f"image_{img_file.split('.')[0]}*")
            matched_files = glob.glob(pattern)
            if len(matched_files) > num_of_images:
                # Choose random N images among them
                gen_imgs_path = random.sample(matched_files, num_of_images)
            else:
                gen_imgs_path = matched_files

            gen_images = []
            for gen_img_path in gen_imgs_path:
                try:
                    gen_img = Image.open(gen_img_path).convert('RGB')
                    gen_images.append(gen_img)
                except Exception as e:
                    print(f"Failed to open {gen_img_path}: {e}")
        else:
            gen_images = None

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", qu_raw)

        if convis_decoding:
            for gen_img in gen_images:
                if model_name == "mplug-owl2":
                    max_edge = max(gen_img.size)
                    gen_img.resize((max_edge, max_edge))
            gen_images_tensor = [vis_processors["eval"](gen_img) for gen_img in gen_images]
            if model_name == "mplug-owl2":
                gen_images_tensor = [torch.tensor(gen_image_tensor["pixel_values"][0]).unsqueeze(0) for gen_image_tensor in
                                     gen_images_tensor]
            else:
                gen_images_tensor = [gen_image_tensor.unsqueeze(0) for gen_image_tensor in gen_images_tensor]
            gen_images = torch.cat(gen_images_tensor, dim=0)
            gen_images = gen_images.to(device, dtype=torch.half)
        else:
            gen_images = None

        with torch.inference_mode():
            with torch.no_grad():
                _, out, _ = model.generate(
                    {"image": norm(image), "prompt":qu, "img_path": image_path},
                    use_nucleus_sampling=sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=args.beam,
                    max_new_tokens=max_new_tokens,
                    output_attentions=True,
                    beam_search=beam_search,
                    convis_decoding=convis_decoding,
                    # ConVis
                    gen_images=gen_images,
                    convis_alpha=convis_alpha,
                    cutoff_lambda=cutoff_lambda,
                )

        output_text = out[0]
        # Remove new line characters from output_text
        output_text = output_text.replace("\n", " ")
        output_text = output_text.replace("\t", " ")

        sentence_list = output_text.split(".")

        # Filter out sentences containing "unk"
        sentence_filter_list = []
        for sentence in sentence_list:
            if "unk" not in sentence:
                sentence_filter_list.append(sentence)
        output_text = ".".join(sentence_filter_list)

        dict_save = {}
        dict_save["image_id"] = img_file
        dict_save["qu"] = qu_raw
        dict_save["label"] = label
        dict_save["caption"] = output_text

        line_to_write = f'{dict_save["image_id"]}\t{dict_save["qu"]}\t{dict_save["label"]}\t{dict_save["caption"]}\n'

        print("image_path: ", image_path)
        print("caption: ", output_text)

        generated_captions_path = os.path.join(
            base_dir,
            f"{mme_type}.txt",
        )
        with open(generated_captions_path, "a") as f:
            f.write(line_to_write)

