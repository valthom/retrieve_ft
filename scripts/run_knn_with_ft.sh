#!/bin/bash
for split in {6..6}
do
    python main.py \
        --exp_name="knn_with_ft_uncond_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        knn \
        --knn-mode="singleclass" \
        --type="knn_with_ft" \
        --dynamic \
        --fix_missing \
        --embedding="raw"
done