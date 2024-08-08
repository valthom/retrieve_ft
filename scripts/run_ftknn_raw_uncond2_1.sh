#!/bin/bash
for split in {5..8}
do
    python main.py \
        --exp_name="ftknn_raw_uncondfull2_split${split}" \
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
        --eval_interval=1
done