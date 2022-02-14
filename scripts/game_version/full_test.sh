echo Select CUDA_VISIBLE_DEVICES
read cuda_device

array=(20 50 100)
for i in "${array[@]}"
do
    set="uniform_n__128"               ## arbitrary text
    full_set="${set/n_/n_"$i"}"
    CUDA_VISIBLE_DEVICES=$cuda_device python test.py --num_trajectories=100 \
                                                     --test_set=$full_set \
                                                     --test_type=test \
                                                     --test_batch_size=1024 \
                                                     --use_mcts \
                                                     --num_playouts=10 \
                                                     --c_puct=2 \
                                                     --epsilon=0.325 \
                                                     --prob_term=puct \
                                                     --node_value_term=game \
                                                     --node_value_scale -1 1 \
                                                     --experiment_name=mcts_test_game \
                                                     --job_type=full \
                                                     --use_pomo_aug
done