import argparse
import torch
import numpy as np
import random
import os
import json
from main_code.utils.torch_objects import device
from main_code.nets.pomo import PomoNetwork
from main_code.utils.logging.logging import Get_Logger
from main_code.utils.config.config import get_config
from main_code.tester.tsp_tester import TSPTester

def test_multiple(test_set_paths=None):
    pass

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./logs/Saved_TSP100_Model')
    # should not be necessary since config should be saved with the model
    parser.add_argument("--config_path", type=str, default='./configs/default.json')
    # test hyperparmeters -> influence performance
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument('--use_pomo_aug', dest='use_pomo_aug', default=False, action='store_true')
    parser.add_argument("--sampling_steps", type=int, default=1)
    # batchsize only relevant for speed, depends on gpu memory
    parser.add_argument("--test_batch_size", type=int, default=1024)
    # random test set specifications
    parser.add_argument("--test_set_size", type=int, default=1e+4)
    parser.add_argument("--num_nodes", type=int, default=100)
    # saved test set path
    parser.add_argument("--test_set_path", type=str, default=None)
    # save options
    parser.add_argument("--save_dir", type=str, default='./results')
    parser.add_argument("--save_folder_name", type=str, default='test')
    opts = parser.parse_known_args()[0]
    
    # set seeds for reproducibility 
    np.random.seed(37)
    random.seed(37)
    torch.manual_seed(37)

    # get config
    config = get_config(opts.config_path)
    # update config based on provided bash arguments
    config.update({'TEST_BATCH_SIZE': opts.test_batch_size})
    if opts.test_set_path is None:
        config.update({'TSP_SIZE': opts.num_nodes})
    

    # Init logger
    logger, result_folder_path = Get_Logger(opts.save_dir, opts.save_folder_name)

    # Load Model
    actor_group = PomoNetwork(config).to(device)
    actor_model_save_path = f'{opts.model_path}/ACTOR_state_dic.pt'
    actor_group.load_state_dict(torch.load(actor_model_save_path, map_location="cuda:0"))
    
    # log model info
    logger.info('==============================================================================')
    logger.info('==============================================================================')
    logger.info(f'  <<< MODEL: {actor_model_save_path} >>>')

    tester = TSPTester(logger, 
                       num_trajectories=opts.num_trajectories,                 
                       num_nodes=config.TSP_SIZE,
                       num_samples=opts.test_set_size, 
                       sampling_steps=opts.sampling_steps, 
                       use_pomo_aug=opts.use_pomo_aug,
                       test_set_path=opts.test_set_path,
                       test_batch_size=config.TEST_BATCH_SIZE)

    tester.test(actor_group)
    tester.save_results(file_path=f'{result_folder_path}/result.json')