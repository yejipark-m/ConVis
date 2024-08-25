import os
import json
import argparse
import sys
import copy
import re
sys.path.append(".")
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")

parser.add_argument(
    "-c",
    "--caption-path",
    type=str,
    required=True,
    help="Path to the generated captions",
)
parser.add_argument(
    "--annotation_path",
    type=str,
    required=True,
    help="Path to the annotation json"
)

args = parser.parse_known_args()[0]

caption_path = args.caption_path
annotation_path = args.annotation_path


caption_file_path = (
    f"{annotation_path}/captions_val2014.json"
)
annotation_file_path = (
    f"{annotation_path}/captions_val2014.json"
)

with open(annotation_file_path, "r") as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

coco = COCO(caption_file_path)

print("caption_files", caption_path)

# Construct the full file path
file_path = caption_path

# Process the file (you would insert your processing code here)
# For example, load the JSON, perform operations, and then save the output

print("file_path: ", file_path)

loaded_json = []
with open(file_path, "r") as f:
    # try:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        # print("idx", idx)
        caption_line = json.loads(line)
        if "woodpecker" in file_path:
            # print("caption_line", caption_line)
            pattern_1 = r"\(\[.*?\]\)"
            pattern_2 = r"\(\[.*?\]\;"
            pattern_3 = r"\[.*?\]\;"
            caption_line["caption"] = re.sub(pattern_1, '', caption_line["caption"])
            caption_line["caption"] = re.sub(pattern_2, '', caption_line["caption"])
            caption_line["caption"] = re.sub(pattern_3, '', caption_line["caption"])
            # print("caption_line", caption_line)
            # input()
        loaded_json.append(caption_line)
    # except:
    #     continue

# Check for duplicate image_id entries and remove them
unique_image_ids = set()
i = 0
while i < len(loaded_json):
    image_id = loaded_json[i]["image_id"]
    if image_id in unique_image_ids:
        loaded_json.pop(i)
    else:
        unique_image_ids.add(image_id)
        i += 1

formulated_output_dict = {}
# overall result
all_overall_scores = defaultdict(list)
# imgToEval per image result
img_to_eval_dict = {}
good = copy.deepcopy(loaded_json)
loaded_json = []
loaded_json = good
# to save memory, load 10 captions at a time
for start_idx in tqdm(
    range(0, len(loaded_json), 10), desc="Generating CHAIR Input"
):
    # define the current iteration end index
    end_idx = min(start_idx + 10, len(loaded_json))
    coco_res = coco.loadRes(
        loaded_json[start_idx:end_idx],
    )

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()

    coco_eval.evaluate()

    # coco_eval.eval = {}
    # keep track of the overall scores
    for metric, score in coco_eval.eval.items():
        all_overall_scores[metric].append(score)

    # imgToEval per image result
    for i, cur_img_id in enumerate(coco_res.getImgIds()):
        cur_eval_dict = coco_eval.evalImgs[i]
        # add caption to the eval dict
        cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
        img_to_eval_dict[cur_img_id] = cur_eval_dict

# overall result
overall_dict = {}
for metric, score in all_overall_scores.items():
    overall_dict[metric] = np.mean(score)
formulated_output_dict["overall"] = overall_dict
formulated_output_dict["imgToEval"] = img_to_eval_dict

# Construct the output file name by replacing the ending
output_file_name = caption_path.replace("_generated_captions.json", "_chair.json")
output_file_path = output_file_name

# Save the processed data to the new file
with open(output_file_path, "w") as f_out:
    json.dump(
        formulated_output_dict, f_out
    )  # Assuming processed_data is the result of your processing

print("output_file_path: ", output_file_path)
