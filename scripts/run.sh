#!/bin/bash
python main.py \
    --exp_name="ftknn_select" \
    ft \
    --dynamic \
    --fix_missing \
    --embedding="layer1"

python main.py \
    --exp_name="ftknn_select_raw" \
    ft \
    --dynamic \
    --fix_missing \
    --embedding="raw"