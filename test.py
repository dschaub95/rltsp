

import numpy as np
import time
import argparse
import torch
from utils.torch_objects import device, Tensor, LongTensor
from environment.environment import GroupEnvironment
from nets.pomo import ACTOR
from utils.utils import Average_Meter
from utils.logging import Get_Logger
from environment.graph_transformer import augment_xy_data_by_8_fold
from configs.config import get_config


def test(config, 
         model_path='./logs/Saved_TSP100_Model',
         save_dir='./results', 
         save_folder_name='test',
         test_set_size=10000,
         test_batch_size=1024,
         num_trajectories=1,
         num_nodes=None,
         use_augmentation=False):
    
    actor_model_save_path = f'{model_path}/ACTOR_state_dic.pt'
    
    if test_batch_size is None:
        test_batch_size = config.TEST_BATCH_SIZE
    
    if num_nodes is not None:
        config.update({'TSP_SIZE': num_nodes})

    num_trajectories = np.clip(num_trajectories, 1, num_nodes)
    
    # Make Log File
    logger, result_folder_path = Get_Logger(save_dir, save_folder_name)

    print(result_folder_path)

    # Load Model
    actor_group = ACTOR(config).to(device)

    actor_group.load_state_dict(torch.load(actor_model_save_path, map_location="cuda:0"))
    actor_group.eval()

    logger.info('==============================================================================')
    logger.info('==============================================================================')
    log_str = '  <<< MODEL: {:s} >>>'.format(actor_model_save_path)
    logger.info(log_str)

    eval_dist_AM_0 = Average_Meter()

    logger.info('===================================================================')
    logger.info(f'Number of considered trajectories: {num_trajectories}')

    timer_start = time.time()
    logger_start = time.time()

    episode = 0
    while episode < test_set_size:
        # generate a batch of random test samples (potentialy augmented)
        # increment number of samples 
        if use_augmentation:
            # provide exactly one batch of samples
            seq = Tensor(np.random.rand(int(np.floor(test_batch_size / 8)), config.TSP_SIZE, 2))
            seq = augment_xy_data_by_8_fold(seq)
            batch_s = seq.size(0)
            episode = episode + batch_s / 8
        else:
            seq = Tensor(np.random.rand(test_batch_size, config.TSP_SIZE, 2))
            batch_s = seq.size(0)
            episode = episode + batch_s

        with torch.no_grad():
            
            env = GroupEnvironment(seq, config.TSP_SIZE)
            group_s = num_trajectories
            group_state, reward, done = env.reset(group_size=group_s)
            actor_group.reset(group_state)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            while not done:
                actor_group.update(group_state)
                action_probs = actor_group.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)
        
        if use_augmentation:
            # reshape result reduce to most promising trajectories for each sampled graph
            reward = torch.reshape(reward, (8, -1, num_trajectories))
            reward, _ = reward.max(dim=0)
        # the max does not do anything if the trajectory has size 1
        max_reward, _ = reward.max(dim=-1)
        eval_dist_AM_0.push(-max_reward)  # reward was given as negative dist
        
        # in case of augmentation select 
        if (time.time()-logger_start > config.LOG_PERIOD_SEC) or (episode >= test_set_size):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            percent = np.round((episode / test_set_size) * 100, 1)
            log_str = f"Ep:{int(episode)} ({percent}%)  T:{timestr}  avg.dist:{eval_dist_AM_0.peek()}"
            # log_str = 'Ep:{:07d}({:5.1f}%)  T:{:s}  avg.dist:{:f}'\
            #     .format(episode, episode/test_set_size*100,
            #             timestr, eval_dist_AM_0.peek())
            logger.info(log_str)
            logger_start = time.time()
        
        
    logger.info('---------------------------------------------------')
    logger.info('average = {}'.format(eval_dist_AM_0.result()))
    logger.info('---------------------------------------------------')
    logger.info('---------------------------------------------------')

def test_multiple(test_set_paths=None):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./logs/Saved_TSP100_Model')
    parser.add_argument("--config_path", type=str, default='./configs/default.json')
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument('--use_augmentation', dest='use_augmentation', default=False, action='store_true')
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--save_folder_name", type=str, default='test')
    # random test set specifications
    parser.add_argument("--test_set_size", type=int, default=1e+4)
    parser.add_argument("--num_nodes", type=int, default=100)
    opts = parser.parse_known_args()[0]
    
    # get config
    config = get_config(opts.config_path)
    
    test(config, 
         model_path=opts.model_path,
         num_trajectories=opts.num_trajectories,
         use_augmentation=opts.use_augmentation,
         test_batch_size=opts.test_batch_size,
         save_folder_name=opts.save_folder_name,
         test_set_size=opts.test_set_size,
         num_nodes=opts.num_nodes)