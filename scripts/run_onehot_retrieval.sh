#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="ftknn_onehot_retrieval_3e-5_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        ft \
        --lr=3e-5 \
        --num_epochs=21 \
        --dynamic \
        --fix_missing \
        --onehot_retrieval \
        --embedding="raw"
done