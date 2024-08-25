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
from PIL import Image

import glob
import json

from pope_loader import POPEDataSet
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

POPE_PATH = {
    "random": "pope_random.json",
    "popular": "pope_pop.json",
    "adversarial": "pope_adv.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--pope_type", type=str, help="model")
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
    parser.add_argument("-m", "--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_false",
        dest="verbosity",
        default=True,
        help="Verbosity. Default: True.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    ### ConVis Argument Starts ###
    parser.add_argument(
        "--convis_alpha", type=float, default=0.1, help="Weight for contrast"
    )
    parser.add_argument("--num_of_images", type=int, default=4, help="Number of used generated images.")
    parser.add_argument(
        "--cutoff_lambda", type=float, default=None
    )
    ### ConVis Argument Ends ###

    parser.add_argument(
        "--debugger",
        type=int,
        default=0,
        help="0 print no debugging output; 1 only print hallucination correction; 2 print all the debugging output.",
    )

    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=500,
        help="Number of images to build POPE questions. Default is 500.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of positive/negative objects to be sampled. Default is 3.",
    )

    parser.add_argument(
        "--exp_id",
        type=str,
        help="To record results.json with exp_id=current_time",
    )

    args = parser.parse_args()
    return args


def setup_seeds(image_seed, model_seed):
    random.seed(image_seed)
    np.random.seed(image_seed)
    torch.manual_seed(model_seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    fb = (1 + 0.2*0.2) * (precision * recall) / (0.2*0.2 * precision + recall)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("FB_0.2 score: {}".format(fb))
    print("Yes ratio: {}".format(yes_ratio))

    return acc, precision, recall, f1, fb


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            pred_list.append(0)
        else:
            pred_list.append(1)

    return pred_list


def main():
    args = parse_args()
    device_map = {'device_map': 'auto'}

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    decoding_strategy = args.decoder
    sample = args.sample
    temperature = args.temperature
    top_p = args.top_p
    setup_seeds(args.image_seed, args.model_seed)
    device = 'cuda'
    model_name = args.model
    verbosity = args.verbosity
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_samples = args.num_samples
    num_images = args.num_images
    data_path = args.data_path
    num_beams = args.beam
    max_new_tokens = args.max_new_tokens
    convis_alpha = args.convis_alpha
    cutoff_lambda = args.cutoff_lambda
    num_of_images = args.num_of_images
    exp_id = args.exp_id

    # ========================================
    #             Model Initialization
    # ========================================
    print("Initializing Model")

    model_config = cfg.model_cfg
    model_config.device_8bit = device_map
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    if model_name in {"minigpt4", "instructblip"}:
        model = model.to(device)
    model.eval()
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
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

    if verbosity:
        print("\ndecoding strategy: ", decoding_strategy)
        print("backbone model_name: ", args.model)

    print("Done!")

    question_path = args.pope_path
    # load all the POPE questions
    all_pope_questions = [json.loads(q) for q in open(question_path, "r")]
    if verbosity:
        print(
            f"\nLoaded {len(all_pope_questions)} POPE questions from {question_path}."
        )
    # sanity check
    if len(all_pope_questions) != num_images * num_samples * 2:
        raise ValueError(
            f"Number of POPE questions loaded from {question_path} is not equal to {num_images * num_samples * 2}."
        )

    generated_image_file_path = args.images_path + "/"+ model_name + f"/num_steps_{args.diff_num_step}/"

    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=question_path, data_path=data_path, trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print("load data finished")

    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        if model_name == 'mplug-owl2':
            image = torch.tensor(image["pixel_values"][0]).to(device, dtype=torch.half)
        qu = data["query"]
        qu = [q + " Please answer yes or no." for q in qu]
        label = data["label"]
        image_path = data["image_path"]

        image_id = image_path[0].split("/")[-1].split(".")[0].split("_")[-1].lstrip("0")
        label_list = label_list + list(label)

        if convis_decoding:
            pattern = generated_image_file_path + f"image_{image_id}*.png"
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
        qu = [template.replace("<question>", q) for q in qu][0]

        image = image.to(device)
        label = torch.Tensor(label).to(device)

        if convis_decoding:
            for gen_img in gen_images:
                if model_name == "mplug-owl2":
                    max_edge = max(gen_img.size)
                    gen_img.resize((max_edge, max_edge))
            gen_images_tensor = [vis_processors["eval"](gen_img) for gen_img in gen_images]
            if model_name == "mplug-owl2":
                gen_images_tensor = [torch.tensor(gen_image_tensor["pixel_values"][0]).unsqueeze(0)
                                     for gen_image_tensor in gen_images_tensor]
            else:
                gen_images_tensor = [gen_image_tensor.unsqueeze(0) for gen_image_tensor in gen_images_tensor]
            gen_images = torch.cat(gen_images_tensor, dim=0)
            gen_images = gen_images.to(device, dtype=torch.half)
        else:
            gen_images = None

        print("image_path", image_path)

        with torch.inference_mode():
            with torch.no_grad():
                _, out, _ = model.generate(
                    {"image": image, "prompt": qu, "image_id": image_id},
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
                if convis_decoding:
                    out = [out[0]]
                pred_list = recorder(out, pred_list)

        output_text = out[0]
        print("output text", output_text)
        sentence_list = output_text.split(".")

        sentence_filter_list = []
        for sentence in sentence_list:
            if "unk" not in sentence:
                sentence_filter_list.append(sentence)
        output_text = ".".join(sentence_filter_list)


    print(
        "[{}, {}]===============================================".format(
            args.scale_factor, args.num_attn_candidates
        )
    )
    if len(pred_list) != 0:
        acc, precision, recall, f1, fb = print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        acc, precision, recall, f1, fb = print_acc(pred_list_s, label_list)

    result = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "FB_0.2 Score": fb,
    }


if __name__ == "__main__":
    main()
