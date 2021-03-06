echo Select CUDA_VISIBLE_DEVICES
read cuda_device

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=3 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=5 \
                                                 --c_puct=7.5

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
                                                 --num_playouts=20 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=30 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=50 \
                                                 --c_puct=7.5

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                 --test_set=uniform_n_20_128 \
                                                 --test_batch_size=1024 \
                                                 --use_mcts \
                                                 --num_playouts=100 \
                                                 --c_puct=7.5



