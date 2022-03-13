import numpy as np
import argparse
import sys
import os

sys.path.insert(1, os.getcwd())
from main_code.utils.data.data_sets import DiskTSPTestSet
from main_code.utils.data.utils import sample_and_save_subset


if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="./data/test_sets/fu_et_al_n_100_10000"
    )
    parser.add_argument(
        "--save_path", type=str, default="./data/test_sets/fu_et_al_sample_n_100_128"
    )
    parser.add_argument("--sample_size", type=int, default=128)
    opts = parser.parse_known_args()[0]
    dataset_path = opts.dataset_path
    dataset = DiskTSPTestSet(dataset_path, use_pomo_aug=False, sampling_steps=1)
    sample_and_save_subset(
        dataset, save_path=opts.save_path, sample_size=opts.sample_size
    )
