echo Select CUDA_VISIBLE_DEVICES
read cuda_device


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=5.0

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug \
                                                 --use_mcts \
                                                 --num_playouts=20 \
                                                 --c_puct=5.0

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug \
                                                 --use_mcts \
                                                 --num_playouts=30 \
                                                 --c_puct=5.0

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug \
                                                 --use_mcts \
                                                 --num_playouts=50 \
                                                 --c_puct=5.0

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=20 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_pomo_aug \
                                                 --use_mcts \
                                                 --num_playouts=100 \
                                                 --c_puct=5.0



