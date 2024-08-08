#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="pfknn_uncond_sqrt8_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        knn \
        --knn-mode="singleclass" \
        --context_length=1000 \
        --dynamic_small \
        --fix_missing
done