echo Select CUDA_VISIBLE_DEVICES
read cuda_device

CUDA_VISIBLE_DEVICES=$cuda_device python test.py \
--model_path=./results/saved_models/saved_tsp100_model \
--c_puct=2 \
--epsilon=0.6 \
--node_value_scale=[0,1] \
--node_value_term=smooth \
--num_parallel=8 \
--num_playouts=10 \
--num_trajectories=100 \
--prob_term=puct \
--test_set=fu_et_al_sample_n_100_128 \
--test_type=test \
--use_mcts=1 \
--use_pomo_aug=1 \
--virtual_loss=20 \
--experiment_name=tsp100_mcts_test

CUDA_VISIBLE_DEVICES=$cuda_device python test.py \
--model_path=./results/saved_models/saved_tsp100_model \
--c_puct=0.5 \
--epsilon=0.6 \
--node_value_scale=[0,1] \
--node_value_term=smooth \
--num_parallel=8 \
--num_playouts=10 \
--num_trajectories=100 \
--prob_term=puct \
--test_set=fu_et_al_sample_n_100_128 \
--test_type=test \
--use_mcts=1 \
--use_pomo_aug=1 \
--virtual_loss=20 \
--experiment_name=tsp100_mcts_test