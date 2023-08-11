#!/bin/sh

python benign.py \
    --use_org_node_attr \
    --train_verbose \
    --save_clean_model \
    --clean_model_save_path output

python attack.py \
    --use_org_node_attr \
    --train_verbose \
    --target_class 0 \
    --bkd_size 3 \
    --bkd_num_pergraph 1 \
    --pn_rate 1 \
    --bkd_gratio_train 0.02 \
    --bkd_gratio_test 0.5 \
    --train_epochs 20
