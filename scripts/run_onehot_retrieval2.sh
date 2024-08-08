#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="ftknn_onehot_retrieval_q2k_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        ft \
        --num_epochs=21 \
        --dynamic \
        --fix_missing \
        --onehot_retrieval \
        --train_query_length=2000 \
        --embedding="raw"
done