#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="test_pfknn_uncond_c100_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        knn \
        --knn-mode="singleclass" \
        --context_length=100 \
        --dynamic \
        --fix_missing
done