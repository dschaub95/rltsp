echo Select CUDA_VISIBLE_DEVICES
read cuda_device

array=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 1)
for i in "${array[@]}"
do
	CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                     --test_set=uniform_n_20_128 \
                                                     --test_type=valid \
                                                     --test_batch_size=1024 \
                                                     --use_mcts \
                                                     --num_playouts=10 \
                                                     --c_puct=2 \
                                                     --epsilon=$i \
                                                     --prob_term=puct \
                                                     --node_value_term=default \
                                                     --node_value_scale -1 1 \
                                                     --experiment_name=mcts_test_default \
                                                     --job_type=epsilon
done
