#!/bin/bash

for benchmark in image_generation/*; do
  if [ -d "$benchmark" ]; then
    benchmark_name=$(basename "$benchmark")

    for model in "$benchmark"/*; do
      if [ -d "$model" ]; then
        model_type=$(basename "$model")

        if [ "$benchmark_name" == "MME" ]; then

          for mme_folder in "$model"/*; do
            if [ -d "$mme_folder" ]; then
              mme_type=$(basename "$mme_folder")

              for file in "$mme_folder"/*; do
                if [ -f "$file" ]; then
                  caption_path="$file"

                  python run_scripts/image_generation.py --model_type "$model_type" --benchmark_name "$benchmark_name" --mme_type "$mme_type" --caption_path "$caption_path" --output_path generated_images
                fi
              done
            fi
          done
        else

          for file in "$model"/*; do
            if [ -f "$file" ]; then
              caption_path="$file"

              mme_type=""

              python run_scripts/image_generation.py --model_type "$model_type" --benchmark_name "$benchmark_name" --mme_type "$mme_type" --caption_path "$caption_path" --output_path generated_images
            fi
          done
        fi

      fi
    done
  fi
done
