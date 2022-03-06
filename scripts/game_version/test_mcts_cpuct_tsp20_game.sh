echo Select CUDA_VISIBLE_DEVICES
read cuda_device

array=(1 2 3 4 5 6 7 8)
for i in "${array[@]}"
do
	CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=1 \
                                                     --test_set=uniform_n_20_128 \
                                                     --test_type=valid \
                                                     --test_batch_size=1024 \
                                                     --use_mcts \
                                                     --num_playouts=10 \
                                                     --c_puct=$i \
                                                     --epsilon=0.9 \
                                                     --prob_term=puct \
                                                     --node_value_term=game \
                                                     --node_value_scale -1 1 \
                                                     --experiment_name=mcts_test_game \
                                                     --job_type=playouts
done
