import argparse
import glob
import os
import sys

sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from PIL import Image
import json

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


def setup_seeds():
    cudnn.benchmark = False
    cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="convis",
    help="Decoding strategy to use. Default is ours 'convis'.",
)
parser.add_argument("-m", "--max_new_tokens", type=int, default=512)
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
    "--diff_num_step",
    type=int,
    default=1,
    help="diffusion num_steps for image generation"
)

parser.add_argument("-b", "--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1)

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


### ConVis Argument Starts ###
parser.add_argument(
    "--convis_alpha", type=float, default=1.0, help="Weight for contrast"
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

setup_seeds()

device = 'cuda'

verbosity = args.verbosity
sample = args.sample
top_p = args.top_p
temperature = args.temperature
data_path = args.data_path
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


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("num_beams: ", num_beams)
    print(vis_processors["eval"].transform)


img_path_dir = args.data_path
img_path_list = os.listdir(img_path_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
generated_captions_path = output_dir + f"/{decoding_strategy}_generated_caption.json"

for image_path in img_path_list:
    if '_t2i' in image_path:
        continue
    img_save = {}
    img_save["image_path"] = image_path
    img_file_path = f"{img_path_dir}/{image_path}"
    if convis_decoding:
        img_name, _ = os.path.splitext(image_path)
        pattern = img_path_dir + '/' + img_name + f"_t2i*"
        gen_images_path = glob.glob(pattern)
        gen_images = [Image.open(gen_image_path).convert("RGB") for gen_image_path in gen_images_path]
    raw_image = Image.open(img_file_path).convert("RGB")


    if model_name == "mplug-owl2":
        max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
        image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([image], model.image_processor)
        image = image_tensor.to(device='cuda', dtype=torch.float16)
    else:
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device='cuda')

    qu = "Please describe this image in detail."
    # qu = "Generate a one sentence caption of the image."
    # qu = "Generate a short caption of the image."
    # qu = "What is the man holding in his hand?"
    # qu = "generate a one sentence caption of the image"

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

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
                {"image": norm(image), "prompt":qu, "img_path": img_file_path},
                use_nucleus_sampling=args.sample, 
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                beam_search=beam_search,
                convis_decoding=convis_decoding,
                # ConVis
                gen_images=gen_images,
                convis_alpha=convis_alpha,
                cutoff_lambda=cutoff_lambda,
            )

    output_text = out[1]

    print("caption: ", output_text)
    img_save["caption"] = output_text
    
    print("Done!")
    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")

