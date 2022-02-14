echo Select CUDA_VISIBLE_DEVICES
read cuda_device

# TSP20
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --use_pomo_aug \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --use_pomo_aug \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=20 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --use_pomo_aug \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=30 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                 --use_pomo_aug \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=40 \
                                                 --c_puct=7.5















