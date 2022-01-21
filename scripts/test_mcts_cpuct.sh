echo Select CUDA_VISIBLE_DEVICES
read cuda_device


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=0.1

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=0.75

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=1.25

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=2.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=5.0

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --c_puct=10.0                                    