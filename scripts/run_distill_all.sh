#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="distill2_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        dist \
        --num_epochs=100 \
        --eval_interval=5 \
        --batch_size=1024 \
        --context_length=1000 \
        --early_stopping_rounds=100 \
        --early_stopping_metric="negloss"
done