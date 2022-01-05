import argparse
import torch
import numpy as np
import random
import wandb
from main_code.utils.torch_objects import device
from main_code.nets.pomo import PomoNetwork
from main_code.agents.policy_agent import PolicyAgent
from main_code.agents.mcts_agent import MCTSAgent
from main_code.utils.logging.logging import get_test_logger
from main_code.utils.config.config import Config
from main_code.tester.tsp_tester import TSPTester

def main():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./logs/train/Saved_TSP100_Model')
    # should not be necessary since config should be saved with the model
    parser.add_argument("--config_path", type=str, default='./configs/default.json')
    # test hyperparmeters -> influence performance
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument('--use_pomo_aug', dest='use_pomo_aug', default=False, action='store_true')
    parser.add_argument("--sampling_steps", type=int, default=1)
    parser.add_argument("--use_mcts", dest='use_mcts', default=False, action='store_true')
    # batchsize only relevant for speed, depends on gpu memory
    parser.add_argument("--test_batch_size", type=int, default=1024)
    # random test set specifications
    parser.add_argument("--random_test", dest='random_test', default=False, action='store_true')
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--tsp_type", type=str, default='uniform') # later add clustered
    # saved test set
    parser.add_argument("--test_set", type=str, default='uniform_n_20_10000')
    # save options
    parser.add_argument("--save_dir", type=str, default='./logs/test')
    opts = parser.parse_known_args()[0]
    return opts

if __name__ == "__main__":
    opts = parse_args()
    
    # set seeds for reproducibility 
    np.random.seed(37)
    random.seed(37)
    torch.manual_seed(37)

    # get config
    config = Config(config_json=opts.config_path, restrictive=False)
    # create new test subconfig
    test_config = Config(config_class=opts, restrictive=False)
    config.test = test_config
    # update config based on provided bash arguments
    # check if test shall be random then skip next step
    if test_config.random_test:
        test_config.test_set_path = None
    else:
        # check whether test set exists if not throw error
        test_config.test_set_path = f'./data/test_sets/{opts.test_set}'
    
    # adjust settings for mcts
    if test_config.use_mcts:
        test_config.num_trajectories = 1
        test_config.test_batch_size = 1
        test_config.sampling_steps = 1
        test_config.use_pomo_aug = False

    # Init logger
    logger, result_folder_path = get_test_logger(test_config)
    # save config to log folder
    config.to_yaml(f'{result_folder_path}/config.yml', nested=True)
    
    # Load Model
    actor_group = PomoNetwork(config).to(device)
    actor_model_save_path = f'{opts.model_path}/ACTOR_state_dic.pt'
    actor_group.load_state_dict(torch.load(actor_model_save_path, map_location=device))
    
    # select the agent
    if test_config.use_mcts:
        agent = MCTSAgent(actor_group)
    else:
        agent = PolicyAgent(actor_group)

    # log model info
    logger.info('==============================================================================')
    logger.info('==============================================================================')
    logger.info(f'  <<< MODEL: {actor_model_save_path} >>>')
    # init tester
    tester = TSPTester(logger, 
                       num_trajectories=test_config.num_trajectories,                 
                       num_nodes=test_config.num_nodes,
                       num_samples=test_config.num_samples, 
                       sampling_steps=test_config.sampling_steps, 
                       use_pomo_aug=test_config.use_pomo_aug,
                       test_set_path=test_config.test_set_path,
                       test_batch_size=test_config.test_batch_size)
    # run test
    tester.test(agent)
    # save results
    tester.save_results(file_path=f'{result_folder_path}/result.json')