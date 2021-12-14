import numpy as np
import random
import time
import os
import argparse
import torch 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 

from utils.torch_objects import Tensor, LongTensor, device
from utils.utils import Average_Meter
from utils.data_loader import TSP_DATA_LOADER__RANDOM
from utils.logging import Get_Logger
from environment.environment import GroupEnvironment
from configs.config import get_config

from nets.pomo import ACTOR

def train(config, 
          save_dir='./logs',
          save_folder_name='train'):
    np.random.seed(37)
    random.seed(37)
    torch.manual_seed(37)
    # Make Log File
    logger, result_folder_path = Get_Logger(save_dir, save_folder_name)


    # Save used HYPER_PARAMS
    # hyper_param_filepath = './HYPER_PARAMS.py'
    # hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path) 
    # shutil.copy(hyper_param_filepath, hyper_param_save_path)


    ############################################################################################################
    ############################################################################################################
    # Objects to Use
    actor = ACTOR(config).to(device)
    actor.optimizer = optim.Adam(actor.parameters(), 
                                 lr=config.ACTOR_LEARNING_RATE, 
                                 weight_decay=config.ACTOR_WEIGHT_DECAY)
    actor.lr_stepper = lr_scheduler.StepLR(actor.optimizer, 
                                           step_size=config.LR_DECAY_EPOCH, 
                                           gamma=config.LR_DECAY_GAMMA)

    # GO
    timer_start = time.time()
    best_dist_avg = np.Inf
    for epoch in range(1, config.TOTAL_EPOCH+1):
        
        log_package = {
            'epoch': epoch,
            'timer_start': timer_start,
            'logger': logger        
        }

        #  TRAIN
        #######################################################
        train_one_epoch(config, actor, **log_package)

        #  EVAL
        #######################################################
        dist_avg = validate(config, actor, **log_package)

        #  CHECKPOINT
        #######################################################
        
        checkpoint_epochs = np.arange(1, config.TOTAL_EPOCH+1, 5)
        # only save checkpoint if the performance has improved --> the last checkpoint is always the best
        if epoch in checkpoint_epochs and dist_avg < best_dist_avg:
            best_dist_avg = dist_avg
            checkpoint_folder_path = '{}/CheckPoint_ep{:05d}'.format(result_folder_path, epoch)
            os.mkdir(checkpoint_folder_path)
            model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.state_dict(), model_save_path)
            optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.optimizer.state_dict(), optimizer_save_path)
            lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(checkpoint_folder_path)
            torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)

def train_one_epoch(config, actor_group, epoch, timer_start, logger):

    actor_group.train()

    distance_AM = Average_Meter()
    actor_loss_AM = Average_Meter()

    train_loader = TSP_DATA_LOADER__RANDOM(num_samples=config.TRAIN_DATASET_SIZE, 
                                           num_nodes=config.TSP_SIZE, 
                                           batch_size=config.TRAIN_BATCH_SIZE)

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
            actor_group.update(group_state)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            group_state, reward, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

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
        if (time.time()-logger_start > config.LOG_PERIOD_SEC) or (episode == config.TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  CLoss:{:5f}  Avg.dist:{:5f}' \
                .format(epoch, episode, episode/config.TRAIN_DATASET_SIZE*100,
                        timestr, actor_loss_AM.result(), 0,
                        distance_AM.result())
            logger.info(log_str)
            logger_start = time.time()

    # LR STEP, after each epoch
    actor_group.lr_stepper.step()


eval_result = []

def update_eval_result(old_result):
    global eval_result
    eval_result = old_result

def validate(config, actor_group, epoch, timer_start, logger):

    global eval_result

    actor_group.eval()

    eval_dist_AM = Average_Meter()
    if config.TSP_SIZE == 5:
        raise NotImplementedError
    elif config.TSP_SIZE == 10:
        raise NotImplementedError
    else:
        test_loader = TSP_DATA_LOADER__RANDOM(num_samples=config.TEST_DATASET_SIZE, 
                                              num_nodes=config.TSP_SIZE, 
                                              batch_size=config.TEST_BATCH_SIZE)

    for data in test_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)
        batch_s = data.size(0)

        with torch.no_grad():
            env = GroupEnvironment(data, config.TSP_SIZE)
            group_s = config.TSP_SIZE
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

        max_reward, _ = reward.max(dim=1)
        eval_dist_AM.push(-max_reward)  # reward was given as negative dist

    # LOGGING
    dist_avg = eval_dist_AM.result()
    eval_result.append(dist_avg)

    logger.info('--------------------------------------------------------------------------')
    log_str = '  <<< EVAL after Epoch:{:03d} >>>   Avg.dist:{:f}'.format(epoch, dist_avg)
    logger.info(log_str)
    logger.info('--------------------------------------------------------------------------')
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')
    return dist_avg


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='./configs/tsp20.json')
    parser.add_argument("--save_dir", type=str, default='./logs')
    parser.add_argument("--save_folder_name", type=str, default='train')
    opts = parser.parse_known_args()[0]
    # get config
    config = get_config(opts.config_path)
    train(config, 
          save_dir=opts.save_dir,
          save_folder_name=opts.save_folder_name)
