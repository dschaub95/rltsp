import wandb
import json
import pandas as pd
import requests
import os
import pathlib
from typing import (
    Any,
    Generator,
    IO,
    Optional,
    Union,
)


def get_table_data_from_url(source_url: str, api_key: Optional[str] = None) -> None:
    response = requests.get(source_url, auth=("api", api_key), stream=True, timeout=5)
    response.raise_for_status()
    bytes_list = []
    for data in response.iter_content(chunk_size=1024):
        bytes_list.append(data)
    final_byte_data = b"".join(bytes_list)
    data_dict = json.loads(final_byte_data.decode("utf-8"))
    table_df = pd.DataFrame(data=data_dict["data"], columns=data_dict["columns"])
    return table_df


def get_data_for_run(run, download_tours=False, download_metrics=False, limit=10000):
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    name = run.name
    if download_tours:
        tour_file = [file for file in run.files() if "tours" in str(file)][0]
        tour_df = get_table_data_from_url(tour_file.url, wandb.Api().api_key)
    else:
        tour_df = pd.DataFrame()
    if download_metrics:
        metrics_file = [file for file in run.files() if "run_metrics" in str(file)][0]
        metrics_df = get_table_data_from_url(metrics_file.url, wandb.Api().api_key)
    else:
        metrics_df = pd.DataFrame()
    history_df = pd.DataFrame()
    for point in run.scan_history(page_size=limit):
        df_tmp = pd.DataFrame(point, index=[0])
        history_df = pd.concat([history_df, df_tmp], ignore_index=True)
    data_dict = {
        "name": name,
        "config": config,
        "summary": summary,
        "history": history_df,
        "tour_data": tour_df,
        "run_metrics": metrics_df,
    }
    return data_dict


def extract_data_from_sweep(sweep_path, download_tours=True, download_metrics=True):
    # restrict which data is downloaded
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    sweep_config = sweep.config
    runs = sweep.runs
    sweep_data = [
        get_data_for_run(run, download_tours, download_metrics) for run in runs
    ]
    return sweep_data
