import logging
import os
import datetime
import pytz
import re
from utils.torch_objects import device

tz = pytz.timezone("Europe/Berlin")


def timetz(*args):
    return datetime.datetime.now(tz).timetuple()


def Get_Logger(save_dir, save_folder_name):
    # make_dir
    #######################################################
    prefix = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime("%Y%m%d_%H%M__")
    result_folder_no_postfix = f"{save_dir}/{prefix + save_folder_name}"

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(result_folder_path)  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('{}/log.txt'.format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)
    logger.info(f'Saving logs in: {result_folder_path}')
    logger.info(f'Using device: {device}')
    return logger, result_folder_path


def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = '{}/log.txt'.format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + '[^\n]+', line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command