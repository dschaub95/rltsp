import argparse
import torch
import numpy as np
import random
import wandb
from main_code.utils.torch_objects import device
from main_code.nets.pomo import PomoNetwork
from main_code.agents.policy_agent import PolicyAgent
from main_code.agents.mcts_agent import MCTSAgent, MCTSBatchAgent, MCTS
from main_code.utils.logging.logging import get_test_logger
from main_code.utils.config.config import Config
from main_code.tester.tsp_tester import TSPTester


def main():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./logs/train/Saved_TSP100_Model"
    )
    # should not be necessary since config should be saved with the model
    parser.add_argument("--config_path", type=str, default="./configs/default.json")
    # test hyperparmeters -> influence performance
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument(
        "--use_pomo_aug", dest="use_pomo_aug", default=False, action="store_true"
    )
    parser.add_argument("--sampling_steps", type=int, default=1)
    # parser.add_argument(
    #     "--use_mcts", dest="use_mcts", default=False, action="store_true"
    # )
    parser.add_argument("--use_mcts", type=int, default=0)
    parser.add_argument("--c_puct", type=float, default=7.5)
    parser.add_argument("--epsilon", type=float, default=0.91)
    parser.add_argument("--node_value_scale", type=str, default="[-1,1]")
    parser.add_argument("--expansion_limit", type=int, default=None)
    parser.add_argument("--node_value_term", type=str, default=None)
    parser.add_argument("--prob_term", type=str, default="puct")
    parser.add_argument("--num_playouts", type=int, default=10)
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--virtual_loss", type=int, default=0)
    # batchsize only relevant for speed, depends on gpu memory
    parser.add_argument("--test_batch_size", type=int, default=1024)
    # random test set specifications
    parser.add_argument(
        "--random_test", dest="random_test", default=False, action="store_true"
    )
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument(
        "--tsp_type", type=str, default="uniform"
    )  # later add clustered
    # saved test set
    parser.add_argument("--test_set", type=str, default="uniform_n_20_128")
    parser.add_argument("--test_type", type=str, default="valid")
    # save options
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--job_type", type=str, default=None)
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
        test_config.test_set_path = f"./data/{opts.test_type}_sets/{opts.test_set}"

    # adjust settings for mcts
    if test_config.use_mcts:
        # test_config.num_trajectories = 1
        test_config.test_batch_size = (
            8 if test_config.use_pomo_aug else test_config.sampling_steps
        )

    # Init logger
    logger, result_folder_path = get_test_logger(test_config)
    # save config to log folder
    if test_config.use_mcts:
        # prepare mcts_config and use as direct input for agent
        mcts_config = Config()
        mcts_config.set_defaults(MCTS)
        # overwrite with parse values
        mcts_config.from_class(test_config)
        config.mcts = mcts_config
    config.to_yaml(f"{result_folder_path}/config.yml", nested=True)
    wandb.init(
        config=test_config,
        group=test_config.experiment_name,
        job_type=test_config.job_type,
    )

    # Load Model
    actor_group = PomoNetwork(config).to(device)
    actor_model_save_path = f"{opts.model_path}/ACTOR_state_dic.pt"
    actor_group.load_state_dict(torch.load(actor_model_save_path, map_location=device))

    # select the agent
    if test_config.use_mcts:
        agent = MCTSAgent(actor_group, mcts_config.to_dict(False))
        # agent = MCTSBatchAgent(actor_group, c_puct=test_config.c_puct, n_playout=test_config.num_playouts, num_parallel=test_config.num_parallel)
    else:
        agent = PolicyAgent(actor_group)

    # log model info
    logger.info(
        "=============================================================================="
    )
    logger.info(
        "=============================================================================="
    )
    logger.info(f"  <<< MODEL: {actor_model_save_path} >>>")
    # init tester
    tester = TSPTester(
        logger,
        num_trajectories=test_config.num_trajectories,
        num_nodes=test_config.num_nodes,
        num_samples=test_config.num_samples,
        sampling_steps=test_config.sampling_steps,
        use_pomo_aug=test_config.use_pomo_aug,
        test_set_path=test_config.test_set_path,
        test_batch_size=test_config.test_batch_size,
    )
    # run test
    tester.test(agent)
    # save results
    tester.save_results(file_path=f"{result_folder_path}/result.json")
