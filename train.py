import numpy as np
import random
import time
import os
import math
import json
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb

from main_code.utils.torch_objects import Tensor, LongTensor, device
from main_code.utils.utils import AverageMeter
from main_code.utils.data.data_loader import TSP_DATA_LOADER__RANDOM
from main_code.utils.logging.logging import Get_Logger
from main_code.environment.environment import GroupEnvironment
from main_code.utils.config.config import get_config
from main_code.testing.tsp_tester import TSPTester
from main_code.agents.policy_agent import PolicyAgent

from main_code.nets.pomo import PomoNetwork


def train(config, save_dir="./logs", save_folder_name="train"):
    # Make Log File
    logger, result_folder_path = Get_Logger(save_dir, save_folder_name)
    # Objects to Use
    actor = PomoNetwork(config).to(device)
    actor.optimizer = optim.Adam(
        actor.parameters(),
        lr=config.ACTOR_LEARNING_RATE,
        weight_decay=config.ACTOR_WEIGHT_DECAY,
    )
    actor.lr_stepper = lr_scheduler.StepLR(
        actor.optimizer, step_size=config.LR_DECAY_EPOCH, gamma=config.LR_DECAY_GAMMA
    )

    # GO
    timer_start = time.time()
    best_valid_avg_len = np.Inf
    for epoch in range(1, config.TOTAL_EPOCH + 1):

        log_package = {"epoch": epoch, "timer_start": timer_start, "logger": logger}

        #  TRAIN
        #######################################################
        train_avg_len, actor_loss = train_one_epoch(config, actor, **log_package)

        #  EVAL
        #######################################################
        valid_result = validate_new(config, actor)
        valid_avg_error = valid_result.avg_approx_error
        valid_avg_len = valid_result.avg_length

        # track the best validation performance
        if valid_avg_len < best_valid_avg_len:
            improvement = True
            best_valid_avg_len = valid_avg_len
            best_valid_avg_error = valid_avg_error
        else:
            improvement = False
        log_data = {
            "train/avg_length": train_avg_len,
            "train/actor_loss": actor_loss,
            "valid/avg_length": valid_avg_len,
            "valid/avg_error": valid_avg_error,
            "valid/best_avg_length": best_valid_avg_len,
            "valid/best_avg_error": best_valid_avg_error,
        }
        wandb.log(log_data)
        fill_str = (
            "--------------------------------------------------------------------------"
        )
        logger.info(fill_str)
        log_str = "  <<< EVAL after Epoch:{:03d} >>>   Avg.dist:{:5f}  Avg.error:{:5f}%".format(
            epoch, valid_avg_len, valid_avg_error
        )
        logger.info(log_str)
        logger.info(fill_str)
        # only save checkpoint if the performance has improved --> the last checkpoint is always the best
        if improvement:
            checkpoint_folder_path = "{}/CheckPoint_ep{:05d}".format(
                result_folder_path, epoch
            )
            os.mkdir(checkpoint_folder_path)
            model_save_path = "{}/ACTOR_state_dic.pt".format(checkpoint_folder_path)
            torch.save(actor.state_dict(), model_save_path)
            optimizer_save_path = "{}/OPTIM_state_dic.pt".format(checkpoint_folder_path)
            torch.save(actor.optimizer.state_dict(), optimizer_save_path)
            lr_stepper_save_path = "{}/LRSTEP_state_dic.pt".format(
                checkpoint_folder_path
            )
            torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)
            # save config
            with open(f"{checkpoint_folder_path}/config.json", "w") as f:
                json.dump(config._items, f, indent=2)


def train_one_epoch(config, actor_group, epoch, timer_start, logger):

    actor_group.train()

    distance_AM = AverageMeter()
    actor_loss_AM = AverageMeter()

    train_loader = TSP_DATA_LOADER__RANDOM(
        num_samples=config.TRAIN_DATASET_SIZE,
        num_nodes=config.TSP_SIZE,
        batch_size=config.TRAIN_BATCH_SIZE,
    )

    logger_start = time.time()
    episode = 0
    for data in train_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)

        batch_s = data.size(0)
        episode = episode + batch_s

        # Actor Group Move
        ###############################################
        env = GroupEnvironment(data, config.TSP_SIZE)
        group_s = config.TSP_SIZE
        group_state, reward, done = env.reset(group_size=group_s)
        actor_group.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
        while not done:
            # actor_group.update(group_state)
            action_probs = actor_group.get_action_probabilities(group_state)
            # shape = (batch, group, TSP_SIZE)
            action = (
                action_probs.reshape(batch_s * group_s, -1)
                .multinomial(1)
                .squeeze(dim=1)
                .reshape(batch_s, group_s)
            )
            # shape = (batch, group)
            group_state, reward, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[
                batch_idx_mat, group_idx_mat, action
            ].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat(
                (group_prob_list, chosen_action_prob[:, :, None]), dim=2
            )

        # LEARNING - Actor
        ###############################################
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()

        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        distance_AM.push(-max_reward)  # reward was given as negative dist
        actor_loss_AM.push(group_loss.detach().reshape(-1))

        # LOGGING
        ###############################################
        # if (time.time() - logger_start > config.LOG_PERIOD_SEC) or (
        #     episode == config.TRAIN_DATASET_SIZE
        # ):
        log_episode = (
            math.ceil(config.TRAIN_DATASET_SIZE / (20 * config.TRAIN_BATCH_SIZE))
            * config.TRAIN_BATCH_SIZE
        )
        if episode % log_episode == 0 or episode == config.TRAIN_DATASET_SIZE:
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
            actor_loss_result = actor_loss_AM.result()
            avg_tour_len = distance_AM.result()
            log_str = "Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  CLoss:{:5f}  Avg.dist:{:5f}".format(
                epoch,
                episode,
                episode / config.TRAIN_DATASET_SIZE * 100,
                timestr,
                actor_loss_result,
                0,
                avg_tour_len,
            )
            logger.info(log_str)
            logger_start = time.time()
    # LR STEP, after each epoch
    actor_group.lr_stepper.step()
    return avg_tour_len, actor_loss_result


def validate_new(config, actor_group):
    agent = PolicyAgent(actor_group)
    num_samples = int(config.valid_path.split("_")[-1])
    tester = TSPTester(
        num_trajectories=config.TSP_SIZE,
        num_nodes=config.TSP_SIZE,
        num_samples=num_samples,
        sampling_steps=1,
        use_pomo_aug=False,
        test_set_path=config.valid_path,
        test_batch_size=config.TEST_BATCH_SIZE,
    )
    # run test
    test_result = tester.test(agent)
    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/tsp20.json")
    parser.add_argument("--save_dir", type=str, default="./results/train")
    parser.add_argument("--save_folder_name", type=str, default="train")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    opts = parser.parse_known_args()[0]

    # set seeds
    np.random.seed(37)
    random.seed(37)
    torch.manual_seed(37)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # get config
    config = get_config(opts.config_path)
    wandb.init(
        config=config,
        mode=opts.wandb_mode,
        # group=config.experiment_name,
        job_type="training",
    )
    config = wandb.config
    train(config, save_dir=opts.save_dir, save_folder_name=opts.save_folder_name)
