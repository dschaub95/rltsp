echo Select CUDA_VISIBLE_DEVICES
read cuda_device

# run all the tests
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --use_pomo_aug \
                                                 --test_set=uniform_n_100_10000 \
                                                 --test_batch_size=1024

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --sampling_steps=8 \
                                                 --test_set=uniform_n_100_10000 \
                                                 --test_batch_size=2048

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --sampling_steps=16 \
                                                 --test_set=uniform_n_100_10000 \
                                                 --test_batch_size=2048

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --sampling_steps=32 \
                                                 --test_set=uniform_n_100_10000 \
                                                 --test_batch_size=2048

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --sampling_steps=64 \
                                                 --test_set=uniform_n_100_10000 \
                                                 --test_batch_size=2048
                        