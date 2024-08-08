#!/bin/bash
for split in {4..9}
do
    python main.py \
        --exp_name="ftknn_onehot_raw_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/medium_datasets.csv" \
        ft \
        --num_epochs=21 \
        --dynamic \
        --fix_missing \
        --use_one_hot_emb \
        --embedding="raw"
done