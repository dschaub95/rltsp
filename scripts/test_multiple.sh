#!/bin/bash

# model path should contain the config file
echo Provide model path:
read model_path

echo Select CUDA_VISIBLE_DEVICES
read cuda_device

# run all the tests
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 --use_pomo_augmentation
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 --sampling_steps=32
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 --sampling_steps=16
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 --sampling_steps=8