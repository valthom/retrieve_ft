#!/bin/bash
python main.py \
    --exp_name="ftknn_raw_save_split0" \
    --split=0 \
    --dataset_list_path="data_splits/tabpfn_datasets.csv" \
    ft \
    --num_epochs=21 \
    --dynamic \
    --fix_missing \
    --save_data \
    --splits_evaluated="all" \
    --embedding="raw"