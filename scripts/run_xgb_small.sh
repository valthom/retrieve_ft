#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="xgb_1000_small_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/add_datasets.csv" \
        xgb
done