import argparse
import os
import random
import sys
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

import json

from pycocotools.coco import COCO

import glob
import torch
from PIL import Image

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


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="convis",
    help="Decoding strategy to use. Default is ours 'convis'.",
)

parser.add_argument("--image_seed", type=int, default=42)
parser.add_argument("--model_seed", type=int, default=42)

parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="eval_dataset/val2014/",
    help="data path",
)
parser.add_argument(
    "--annotation_path",
    type=str,
    default="COCO2014/annotations/",
    help="annotation path",
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

parser.add_argument("-n", "--num_samples", type=int, default=500)
parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
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
    default="./log/",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)

### ConVis Argument Starts ###
parser.add_argument(
    "--convis_alpha", type=float, default=1.0, help="Weight for contrast"
)
parser.add_argument("--num_of_images", type=int, default=4, help="Number of used generated images.")
parser.add_argument(
    "--cutoff_lambda", type=float, default=None
)
### ConVis Argument Ends ###

parser.add_argument(
    "--detector",
    type=str,
    default="dino",
    help="Detector type. Default is 'groundingdino'.",
)
parser.add_argument(
    "--debugger",
    type=int,
    default=0,
    help="0 print no debugging output; 1 only print hallucination correction; 2 print all the debugging output.",
)
parser.add_argument("--box_threshold", type=float, default=0.45, help="Box threshold for DINO.")
parser.add_argument(
    "--gt_seg_path",
    type=str,
    default="pope_coco/coco_ground_truth_segmentation.json",
    help="Input json file that contains ground truth objects in the image.",
)
parser.add_argument("--skip_num", type=int, default=0, help="Skip the first skip_num samples.")
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
num_samples = args.num_samples

device = 'cuda'

verbosity = args.verbosity
sample = args.sample
top_p = args.top_p
temperature = args.temperature
data_path = args.data_path
annotation_path = args.annotation_path
output_dir = args.output_dir
num_beams = args.beam
max_new_tokens = args.max_new_tokens
debugger = args.debugger
gt_seg_path = args.gt_seg_path
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

valid_decoding_strategies = [
        "greedy",
        "beam",
        "sample",
        "convis",
    ]


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
    print("num_samples: ", num_samples)
    print("num_beams: ", num_beams)
    print("seed: ", args.image_seed)


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

annotation_file_path = f"{annotation_path}/instances_val2014.json"
caption_file_path = f"{annotation_path}/captions_val2014.json"
generated_image_file_path = args.images_path + "/" + model_name + f"/num_steps_{args.diff_num_step}/"
with open(annotation_file_path, "r") as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(caption_file_path)

img_ids = coco.getImgIds()

sampled_img_ids = random.sample(img_ids, num_samples)

sampled_img_ids = sampled_img_ids[skip_num:]

print("sampled_img_ids", len(sampled_img_ids))

img_files = []
for cur_img_id in sampled_img_ids:
    cur_img = coco.loadImgs(cur_img_id)[0]
    cur_img_path = cur_img["file_name"]
    img_files.append(cur_img_path)

if convis_decoding:
    generated_img_files = {}
    for cur_img_id in sampled_img_ids:
        pattern = generated_image_file_path + f"image_{cur_img_id}*.png"
        matched_files = glob.glob(pattern)

        if len(matched_files) > num_of_images:
            # Choose random N images among them
            selected_files = random.sample(matched_files, num_of_images)
        else:
            selected_files = matched_files

        generated_img_files[cur_img_id] = selected_files

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {
        "name": img_info["file_name"],
        "anns": [],
        "gen_images": []
    }

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )

if convis_decoding:
    for img_id, gen_paths in generated_img_files.items():
        if img_id in img_dict:
            img_dict[img_id]["gen_images"].extend(gen_paths)

base_dir = os.path.join(output_dir, "chair", args.model)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for idx, img_id in tqdm(enumerate(range(len(img_files))), total=len(img_files)):
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])

    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = os.path.join(data_path, img_file)
    raw_image = Image.open(image_path).convert('RGB')

    if convis_decoding:
        gen_imgs_path = img_info.get("gen_images", [])
        gen_images = []
        for gen_img_path in gen_imgs_path:
            try:
                gen_img = Image.open(gen_img_path).convert('RGB')
                gen_images.append(gen_img)
            except Exception as e:
                print(f"Failed to open {gen_img_path}: {e}")

    if model_name == "mplug-owl2":
        max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
        image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([image], model.image_processor)
        image = image_tensor.to(device, dtype=torch.float16)
    elif model_name == "minigpt4" or "instructblip":
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device, dtype=torch.half)
    else:
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)



    qu = "Please describe this image in detail."
    # qu = "Please provide a very detailed description of the image."
    # qu = "Please provide a very long and detailed description of the image."
    # qu = "Generate a one sentence caption of the image."
    # qu = "Generate a short caption of the image."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    if convis_decoding:
        for gen_img in gen_images:
            if model_name == "mplug-owl2":
                gen_img.resize((max_edge, max_edge))
        if model_name == "mplug-owl2":
            gen_images_tensor = [process_images([gen_img], model.image_processor) for gen_img in gen_images]
        else:
            gen_images_tensor = [vis_processors["eval"](gen_img).unsqueeze(0) for gen_img in gen_images]
        gen_images = torch.cat(gen_images_tensor, dim=0)
        gen_images = gen_images.to(device, dtype=torch.half)
    else:
        gen_images = None

    with torch.inference_mode():
        with torch.no_grad():
            _, out, _ = model.generate(
                {"image": image, "prompt": qu, "image_id": img_id},
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
    sentence_list = output_text.split(".")

    sentence_filter_list = []
    for sentence in sentence_list:
        if "unk" not in sentence:
            sentence_filter_list.append(sentence)
    output_text = ".".join(sentence_filter_list)

    img_save["caption"] = output_text

    print("image_path: ", image_path)
    print("caption: ", output_text)

    generated_captions_path = os.path.join(
        base_dir,
        f"{decoding_strategy}_generated_captions.json",
    )

    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")
