export split=0

python main.py \
    --exp_name="vanilla_c1000_timing_split${split}" \
    --split=${split} \
    --dataset_list_path="data_splits/tabpfn_datasets.csv" \
    --timing \
    vanilla \
    --batch_size=512 \
    --context_length=1000

python main.py \
    --exp_name="vanilla_c100_timing_split${split}" \
    --split=${split} \
    --dataset_list_path="data_splits/tabpfn_datasets.csv" \
    --timing \
    vanilla \
    --batch_size=512 \
    --context_length=100

python main.py \
    --exp_name="vanilla_c200_timing_split${split}" \
    --split=${split} \
    --dataset_list_path="data_splits/tabpfn_datasets.csv" \
    --timing \
    vanilla \
    --batch_size=512 \
    --context_length=200

# python main.py \
#     --exp_name="ftknn_timing_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=10000 \
#     --context_length=1000 

# python main.py \
#     --exp_name="ftknn_timing_c100_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=10000 \
#     --context_length=100 \

# python main.py \
#     --exp_name="ftknn_timing_c50_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=10000 \
#     --context_length=50 \

# python main.py \
#     --exp_name="ftknn_timing_c200_all_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=1 \
#     --context_length=200 \

# python main.py \
#     --exp_name="ftknn_timing_c100_all_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=1 \
#     --context_length=100 \

# python main.py \
#     --exp_name="ftknn_timing_c50_all_split${split}" \
#     --split=${split} \
#     --dataset_list_path="data_splits/tabpfn_datasets.csv" \
#     --timing \
#     ft \
#     --knn-mode="singleclass" \
#     --num_epochs=21 \
#     --dynamic \
#     --fix_missing \
#     --save_data \
#     --splits_evaluated="valid" \
#     --embedding="raw" \
#     --eval_interval=1 \
#     --context_length=50 \