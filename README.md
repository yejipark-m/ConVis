
# ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models



[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)

This repository provides the official PyTorch implementation of the following paper:  
> **[ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models](https://arxiv.org/abs/2408.13906)**  
> [Yeji Park](https://yejipark-m.github.io)<sup>†</sup>, [Deokyeong Lee](mailto:plmft@sogang.ac.kr)<sup>†</sup>, [Junsuk Choe](https://sites.google.com/site/junsukchoe/), [Buru Chang](https://sites.google.com/view/buru-chang)  
> Sogang University  
> <sup>†</sup> These authors contributed equally to this work.

<br>

Our implementation is built upon several existing repositories. Specifically, we have borrowed and adapted code from the following sources:

We sincerely appreciate the authors for their foundational work.

- [HALC](https://github.com/BillChan226/HALC): Used as the foundational base for our implementation.

- [LLaVA](https://github.com/haotian-liu/LLaVA): Utilized for evaluating models on the LLaVA-Bench benchmark.

- [HallusionBench](https://github.com/tianyi-lab/HallusionBench): Integrated for evaluating models on the HallusionBench benchmark.


## Table of Contents

- [Installation](#installation)
- [Download Datasets](#download-datasets)
  - [CHAIR / POPE](#chair--pope)
  - [HallusionBench](#hallusionbench)
  - [MME Benchmark](#mme-benchmark)
  - [LLaVA-Bench](#llava-bench)
- [Prepare T2I Generated Images](#prepare-t2i-generated-images)
- [Prepare MLLM Checkpoints](#prepare-mllm-checkpoints)
- [Evaluation](#evaluation)
  - [Running CHAIR evaluation](#running-chair-evaluation)
  - [Running HallusionBench evaluation](#running-hallusionbench-evaluation)
  - [Running POPE evaluation](#running-pope-evaluation)
  - [Running MME evaluation](#running-mme-evaluation)
  - [Running LLaVA-Bench evaluation](#running-llava-bench-evaluation)
- [Demo Playground](#demo-playground)
- [License](#license)


## Installation

We provide the Dockerfile that includes the environment that you need.
This docker image is based on Ubuntu 22.04 and CUDA 12.0.0.

To install, run the following commands to build the environment:

1. Clone the repository locally.
    ```sh
    git clone <current repo>
    ```
2. Build the Docker image.
    ```sh
    docker build -t convis:<your_tag> .
    ```
3. Run the container.
    ```sh
    docker run -itd --name <container name> -v <local repo path>:/root/share/ -p 14352:8888 -p 14353:8889 -p 14354:8890 --shm-size=128G --gpus all -m "128G" --restart=always --ipc=host convis:<your_tag>
    ```
4. Open the container.
    ```sh
    docker exec -it <container name> /bin/bash
    ```

<br>

---

## Download Datasets

### CHAIR / POPE

You have to download [MSCOCO 2014](https://cocodataset.org/#home) dataset for CHAIR / POPE evaluation. Please download and extract it in your data path.

To ensure the dataset is organized correctly, follow the structure below:

```bash
COCO2014/
├── annotations/
│ ├── captions_val2014.json
│ ├── captions_train2014.json
│ ├── instances_train2014.json
│ ├── instances_val2014.json
│ ├── person_keypoints_train2014.json
│ └── person_keypoints_val2014.json
├── train2014/
│ ├── COCO_train2014_000000000001.jpg
│ ├── COCO_train2014_000000000002.jpg
│ └── ...
├── val2014/
│ ├── COCO_val2014_000000000042.jpg
│ ├── COCO_val2014_000000000073.jpg
│ └── ...
└── ...
``` 

### HallusionBench

First, clone the original [HallusionBench](https://github.com/tianyi-lab/HallusionBench.git) repository.
```sh
git clone https://github.com/tianyi-lab/HallusionBench.git
```
Next, download the dataset from the following [official HallusionBench dataset link](https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view) and place it under the HallusionBench directory, maintaining the structure as follows:

```bash
HallusionBench/
└── hallusion_bench/   
```


### MME Benchmark

Please follow the instruction from the ofiicial repository of [MME benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md) to download the dataset.

### LLaVA-Bench

To download the LLaVA-Bench (In-the-Wild), run the following code:

```sh
apt-get install git-lfs && \
git clone https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild && \
cd llava-bench-in-the-wild && \
wget https://github.com/haotian-liu/LLaVA/raw/main/llava/eval/table/rule.json
```

## Prepare T2I Generated Images

Our method requires T2I-generated images during the decoding phases. We provide the captions used to generate the images in our experiments.

### Steps to Download and Set Up T2I Images

1. **Download the Captions and Images**  
   Download the `.zip` file containing the T2I-generated images from the following [link](https://drive.google.com/file/d/1nIkoVyOr3h18Qeu4YMivWwD-7jK-eFcR/view?usp=sharing).

2. **Unzip the File**  
   Unzip the downloaded `.zip` file under the root directory of this repository.

3. **Generate Images**  
   Ensure you are in the root directory of this project. Then, run the following command to generate the images:

   ```sh
   bash image_generation.sh
## Prepare MLLM Checkpoints

Other models automatically download checkpoints from Huggingface when executing the code.
However, the MiniGPT-4 weights needs to be downloaded separately from [official MiniGPT-4 7B pretrained weights for LlaMA-2](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).


Please follow these steps:

1. **Download the checkpoint from link above**:

2. **Generate the folder `model_checkpoints`**:
    - Create a directory named `model_checkpoints` in the current working directory.
    - You can use the following command in your terminal to create the directory:
    ```sh
    mkdir -p ./model_checkpoints
    ```

3. **Move the downloaded checkpoint to the `model_checkpoints` folder**:
    - Move the downloaded file into the newly created `model_checkpoints` directory. You can use the following command:
    ```sh
    mv path/to/downloaded/checkpoint ./model_checkpoints/
    ```

---

# Evaluation

For the sake of brevity and clarity, we have only included instructions specific to executing our method in this README. If you are interested in learning how other methods work, please refer to the [HALC](https://github.com/BillChan226/HALC) README for additional details and arguments.

## Running CHAIR evaluation

There are 3 steps for evaluation on CHAIR benchmark.

1. **Generate the caption with MLLM models using our decoding method**.

<br>   

| Argument            | Example                       | Description                                                                       |
|---------------------|-------------------------------|-----------------------------------------------------------------------------------|
| `--model`           | `llava-1.5`                   | Specify the MLLM model, this codebase supports `minigpt4`, `llava-1.5`, `mPLUG-Owl2`. |
| `--decoder`         | `convis`                      | Choose decoding strategy to use, Default is ours `convis`.                        |
| `--data_path`       | `/path/to/dataset`            | Path to the dataset file or folder, e.g., `COCO2014/val2014`.                     |
| `--annotation_path` | `/path/to/dataset/annotation` | Path to the dataset file or folder, e.g., `COCO2014/annotations`.                 |
| `--output_dir`      | `./generated_captions`        | Directory to save the generated captions.                                         |
| `--images_path`     | `./generated_images/CHAIR`    | Path where the T2I generated images are stored.                                   |

   - Example:

```sh
python run_scripts/caption_generation.py --model llava-1.5 --decoder convis --data_path COCO2014/val2014 --annotation_path COCO2014/annotations --output_dir ./generated_captions --images_path ./generated_images/CHAIR  
```


2. **Generate the caption into CHAIR json file.**


| Argument            | Example                      | Description                                                       |
|---------------------|------------------------------|-------------------------------------------------------------------|
| `-c`                | `path/to/caption`            | Path to the caption json.                                         |
| `--annotation_path` | `/path/to/dataset/annotation` | Path to the dataset file or folder, e.g., `COCO2014/annotations`. |


   - Example:

```sh
python eval/caption_to_chair.py -c ./generated_captions/llava-1.5/convis_generated_captions.json --annotation_path COCO2014/annotations
```
   - The converted CHAIR file will be located in the same folder as the caption file.

   - For your information, converting caption to the CHAIR json file could be time consuming.
If you want to convert more captions at once, consider modifying the value in the following file:

     - **File:** `eval/caption_to_chair.py`
     - **Line:** 141



3. **Evaluate the CHAIR json file.**

   - Example:

```sh
python eval/eval_hallucination.py --metric chair --chair_input_path ./generated_captions/llava-1.5/convis_chair.json --data_dir COCO2014
```


## Running HallusionBench evaluation

There are 2 steps for evaluation on HallusionBench evaluation.

1. **Generate the caption for HallusionBench.**

Arguments are as much as the same with CHAIR.
   - Example:

```sh
python run_scripts/hallusion_eval.py --model llava-1.5 --decoder convis --data_path ./HallusionBench --output_dir ./generated_captions --images_path ./generated_images/HallusionBench
```


2. **Run the GPT-4V evaluation**
Note that you need open-ai API key to evaluate with GPT-4V.
Please write your open-ai API key in 
   - **File:** `eval/utils.py`
   - **Line:** 15

   - Example:
```sh
python eval/hallusion_evaluation.py --model llava-1.5 --decoder convis   
```

## Running POPE evaluation

To evaluate POPE Score,

   
- Example:

```sh
python run_scripts/pope_eval.py --model llava-1.5 --decoder convis --pope_type random --data_path COCO2014/val2014/ --images_path ./generated_images/POPE
```

## Running MME evaluation

Please follow the steps to get MME evaluation results.

1. Generate all the response for all mme_type with below code.

   - Example:
```sh
#!/bin/bash

mme_type_list=("existence" "count" "position" "color" "posters" "celebrity" "scene" "landmark" 
               "artwork" "OCR" "commonsense_reasoning" "numerical_calculation" 
               "text_translation" "code_reasoning")

for mme_type in "${mme_type_list[@]}"
do
    python run_scripts/mme_eval.py --mme_type "$mme_type" --data_path ./MME_benchmark --output_dir ./generated_captions --images_path ./generated_images/MME
done
```

2. Evaluate the responses.

   - Example:
```sh
python eval/MME_score.py --results_dir generated_captions/mme/llava-1.5/convis
```

## Running LLaVA-Bench evaluation

1. Generate response for evaluation.

   - Example:

```sh
python run_scripts/llava_bench_eval.py  --model llava-1.5 --decoder convis --data_path ./LLaVA-Bench --gpu-id 0 --output_dir ./generated_captions --images_path ./generated_images/LLaVA-Bench
```

2. Evaluate with GPT-4
   - Note that you need open-ai API key as same as HallusionBench evaluation.

   - Example:
```sh
python eval/eval_gpt_review_bench.py --question llava-bench-in-the-wild/questions.jsonl --context llava-bench-in-the-wild/context.jsonl --rule llava-bench-in-the-wild/rule.jsonl --answer-list llava-bench-in-the-wild/answers_gpt4.jsonl generated_captions/llava-bench/llava-1.5/convis_generated_captions.json --output generated_captions/llava-bench/llava-1.5/convis_gpt_review.jsonl && \
python eval/summarize_gpt_review.py -f generated_captions/llava-bench/llava-1.5/convis_gpt_review.jsonl
```


## Demo Playground
Try your own image for fun!

1. **Generate the caption first without our decoding method.**

   - Example:
```bash
python run_scripts/demo_inference.py --data_path ./playground --output_dir ./playground -d greedy
```

2. **Generate the image with caption.**

   - Example:
```bash
python run_scripts/image_generation.py --benchmark_name demo --caption_path ./playground/greedy_generated_caption.json --output_path ./playground
```

   - Generated image should be located in the same folder as your own image.
   - Change your generated image file name by appending `_t2i` to the file name you want.

Example:

`your/file/directory/your_filename.jpg`

then generated file name should be

`your/file/directory/your_filename_t2i.png`



3. **Now you can play with your own generated images.**
```bash
python run_scripts/demo_inference.py --data_path [path/to/image/dir] --output_dir [path/to/output] -d convis
```


---

## License

This repository is under [MIT License](LICENSE.md).

