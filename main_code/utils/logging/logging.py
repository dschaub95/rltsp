import logging
import os
import datetime
import pytz
import re
from main_code.utils.torch_objects import device

tz = pytz.timezone("Europe/Berlin")


def timetz(*args):
    return datetime.datetime.now(tz).timetuple()


def check_test_set_exists(test_set_name):
    return os.path.isdir(f"./data/test_sets/{test_set_name}")


def extract_test_set_info(test_set_name):
    num_nodes = test_set_name.split("_")[-2]
    num_samples = test_set_name.split("_")[-1]
    tsp_type = test_set_name.split("_")[0]
    return tsp_type, num_nodes, num_samples


def prepare_test_result_folder(test_config):
    save_dir = test_config.save_dir
    time_id = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    if test_config.test_set_path is not None:
        test_set_name = test_config.test_set_path.split("/")[-1]
        tsp_type, num_nodes, num_samples = extract_test_set_info(test_set_name)
    else:
        test_set_name = "uniform_random"
        num_nodes = test_config.num_nodes
        num_samples = test_config.num_samples
    trajs = test_config.num_trajectories

    if test_config.use_pomo_aug:
        ssteps = 8
    else:
        ssteps = test_config.sampling_steps

    result_folder_no_postfix = (
        f"{save_dir}/{test_set_name}/{time_id}__n_{num_nodes}_{num_samples}_"
        f"traj_{trajs}_ssteps_{ssteps}"
    )
    # add pomo aug identifier
    if test_config.use_pomo_aug:
        result_folder_no_postfix = f"{result_folder_no_postfix}_pomo_aug"

    if test_config.use_mcts:
        result_folder_no_postfix = f"{result_folder_no_postfix}_mcts"

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)
    return result_folder_path


def get_test_logger(config):
    result_folder_path = prepare_test_result_folder(test_config=config)
    os.makedirs(result_folder_path)
    logger = get_logger(result_folder_path)
    return logger, result_folder_path


def get_logger(result_folder_path):
    # Logger
    #######################################################
    logger = logging.getLogger(
        result_folder_path
    )  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler("{}/log.txt".format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)
    logger.info(f"Saving logs in: {result_folder_path}")
    logger.info(f"Using device: {device}")
    return logger


def Get_Logger(save_dir, save_folder_name):
    # make_dir
    #######################################################
    time_id = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime(
        "%Y%m%d_%H%M"
    )
    result_folder_no_postfix = f"{save_dir}/{save_folder_name}_{time_id}"

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(
        result_folder_path
    )  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler("{}/log.txt".format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)
    logger.info(f"Saving logs in: {result_folder_path}")
    logger.info(f"Using device: {device}")
    return logger, result_folder_path


def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = "{}/log.txt".format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + "[^\n]+", line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command
