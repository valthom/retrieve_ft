#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="ftknn_onehot_small_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/add_datasets.csv" \
        ft \
        --num_epochs=21 \
        --dynamic \
        --fix_missing \
        --use_one_hot_emb \
        --embedding="layer1"
done