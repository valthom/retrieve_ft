#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="vanilla_tabpfn_3k_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        vanilla \
        --batch_size=64 \
        --context_length=3000
done