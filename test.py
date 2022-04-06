import argparse
import torch
import numpy as np
import random
import wandb
from main_code.utils.torch_objects import device
from main_code.nets.pomo import PomoNetwork
from main_code.agents.policy_agent import PolicyAgent
from main_code.agents.mcts_agent.mcts_agent import MCTSAgent, MCTSBatchAgent

# from main_code.agents.adaptive_policy_agent_old import AdaptivePolicyAgent

from main_code.agents.adaptive_policy_agent import AdaptivePolicyAgent
from main_code.agents.mcts_agent.mcts import MCTS
from main_code.utils.logging.logging import get_test_logger
from main_code.utils.config.config import Config

# from main_code.testing.tsp_tester import TSPTester

from main_code.testing.tsp_tester_new import TSPTester


def main():
    pass


def parse_adaptlr_args():
    parser = argparse.ArgumentParser()
    # num_epochs=4,
    # batch_size=8,
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--state_space_size", type=int, default=32)
    parser.add_argument("--lr_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_decay_epoch", type=int, default=1)
    parser.add_argument("--lr_decay_gamma", type=float, default=1.0)
    parser.add_argument("--noise_factor", type=float, default=0.0)
    adaptlr_opts = parser.parse_known_args()[0]
    return adaptlr_opts


# different argument parsers for different settings
def parse_mcts_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_puct", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--weight_fac", type=float, default=50)
    parser.add_argument("--node_value_scale", type=str, default="[0,1]")
    parser.add_argument("--expansion_limit", type=int, default=None)
    parser.add_argument("--node_value_term", type=str, default="smooth")
    parser.add_argument("--prob_term", type=str, default="puct")
    parser.add_argument("--num_playouts", type=int, default=10)
    parser.add_argument("--num_parallel", type=int, default=8)
    parser.add_argument("--virtual_loss", type=int, default=20)
    mcts_opts = parser.parse_known_args()[0]
    return mcts_opts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/saved_models/saved_tsp20_model",
        # default="./results/saved_models/saved_tsp50_model",
    )
    # test hyperparmeters -> influence performance
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--use_pomo_aug", type=int, default=0)
    parser.add_argument("--sampling_steps", type=int, default=1)

    # whether to use agent with mcts planning
    parser.add_argument("--use_mcts", type=int, default=0)
    # use agent with adaptive learning
    parser.add_argument("--use_adaptlr", type=int, default=0)

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
    parser.add_argument("--num_workers", type=int, default=2)
    # saved test set
    parser.add_argument("--test_set", type=str, default="fu_et_al_n_20_10000")
    parser.add_argument("--test_type", type=str, default="test")
    # save options
    parser.add_argument("--save_dir", type=str, default="./results")
    # wandb stuff
    parser.add_argument("--job_type", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    opts = parser.parse_known_args()[0]
    return opts


if __name__ == "__main__":
    opts = parse_args()

    # set seeds for reproducibility
    np.random.seed(37)
    random.seed(37)
    torch.manual_seed(37)

    # get config
    config = Config(config_json=f"{opts.model_path}/config.json", restrictive=False)
    # create new test subconfig
    config.test = Config(config_class=opts, restrictive=False)
    # config.test = config.test
    # update config based on provided bash arguments
    # check if test shall be random then skip next step
    if config.test.random_test:
        config.test.test_set_path = None
    else:
        # check whether test set exists if not throw error
        config.test.test_set_path = f"./data/{opts.test_type}_sets/{opts.test_set}"
        # extract some extra info based on the test set name
        config.test.num_samples = int(opts.test_set.split("_")[-1])
        config.test.num_nodes = int(opts.test_set.split("_")[-2])

    # handle arbitrary test batch sizes for various sized inputs
    if config.test.use_pomo_aug:
        config.test.test_batch_size = max(8, config.test.test_batch_size)
    else:
        config.test.test_batch_size = max(config.test.sampling_steps, config.test.test_batch_size)

    # adjust settings for mcts
    if config.test.use_mcts:
        config.test.test_batch_size = (
            8 if config.test.use_pomo_aug else config.test.sampling_steps
        )
        # parse mcts arguments
        mcts_opts = parse_mcts_args()
        # prepare mcts_config and use as direct input for agent
        config.test.mcts = Config()
        # optional if defaults are set in class definiton instead of during parsing
        config.test.mcts.set_defaults(MCTS)
        # overwrite with parse values
        config.test.mcts.from_class(mcts_opts)

    if config.test.use_adaptlr:
        adaptlr_opts = parse_adaptlr_args()
        config.test.adaptlr = Config(config_class=adaptlr_opts, restrictive=False)
        if config.test.adaptlr.batch_size > config.test.adaptlr.state_space_size:
            config.test.adaptlr.batch_size = config.test.adaptlr.state_space_size

    # Init logger
    logger, result_folder_path = get_test_logger(config.test)

    wandb.init(
        config=config.to_dict(),
        mode=config.test.wandb_mode,
        group=config.test.experiment_name,
        job_type=config.test.job_type,
    )
    # wandb cant handle sub configs by itself
    config = Config(config_dict=wandb.config._items, restrictive=False)
    # save config to log folder
    config.to_yaml(f"{result_folder_path}/config.yml", nested=True)

    # Load Model - could be done inside agent
    nnetwork = PomoNetwork(config).to(device)
    model_save_path = f"{opts.model_path}/ACTOR_state_dic.pt"
    nnetwork.load_state_dict(torch.load(model_save_path, map_location=device))
    # select the agent
    if config.test.use_mcts:
        agent = MCTSAgent(nnetwork, config.test.mcts.to_dict(False))
        # agent = MCTSBatchAgent(nnetwork, c_puct=config.test.c_puct, n_playout=config.test.num_playouts, num_parallel=config.test.num_parallel)
    elif config.test.use_adaptlr:
        agent = AdaptivePolicyAgent(nnetwork, **config.test.adaptlr.to_dict(False))
    else:
        agent = PolicyAgent(nnetwork)

    # log model info
    fill_str = (
        "=============================================================================="
    )
    logger.info(fill_str)
    logger.info(fill_str)
    logger.info(f"  <<< MODEL: {model_save_path} >>>")
    # init tester
    tester = TSPTester(
        logger,
        num_trajectories=config.test.num_trajectories,
        num_nodes=config.test.num_nodes,
        num_samples=config.test.num_samples,
        sampling_steps=config.test.sampling_steps,
        use_pomo_aug=config.test.use_pomo_aug,
        test_set_path=config.test.test_set_path,
        test_batch_size=config.test.test_batch_size,
        num_workers=config.test.num_workers,
    )
    # run test
    test_result = tester.test(agent)
    # save results
    tester.save_results(file_path=f"{result_folder_path}/result.json")
