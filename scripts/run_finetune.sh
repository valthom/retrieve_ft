#!/bin/bash
for split in {1..9}
do
    python main.py \
        --exp_name="finetune2_split${split}" \
        --split=${split} \
        --dataset_list_path="data_splits/tabpfn_datasets.csv" \
        ft \
        --type="ft" \
        --num_epochs=100 \
        --eval_interval=5 \
        --batch_size=1024 \
        --batch_size_inf=1024 \
        --context_length=1000 \
        --dynamic
done