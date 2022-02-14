echo Select CUDA_VISIBLE_DEVICES
read cuda_device

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=1 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=2 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=5 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=10 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=15 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=10 \
                                                 --num_parallel=20 \
                                                 --c_puct=7.5