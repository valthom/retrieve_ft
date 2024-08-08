#!/bin/bash
for split in {0..9}
do
    python main.py \
        --exp_name="ftknn_raw_uncondfull2_onehotall_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        ft \
        --knn-mode="singleclass" \
        --num_epochs=21 \
        --dynamic \
        --fix_missing \
        --save_data \
        --splits_evaluated="valid" \
        --embedding="raw" \
        --onehot_retrieval \
        --use_one_hot_emb \
        --eval_interval=1
done