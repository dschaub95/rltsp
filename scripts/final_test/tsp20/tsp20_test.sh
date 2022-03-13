#!/bin/bash

echo Select CUDA_VISIBLE_DEVICES
read cuda_device

# run all the tests

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_type=test \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug=0 \
                                                 --test_set_path=./data/test_sets/fu_et_al_n_20_10000