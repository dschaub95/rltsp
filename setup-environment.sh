#!/usr/bin/env bash -l
# conda create -y --name rltsp python=3.7
# conda activate rltsp
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y pyg -c pyg -c conda-forge
conda install -y numpy scipy cython tqdm scikit-learn matplotlib seaborn tensorboard pandas networkx
conda install -y jupyterlab -c conda-forge
pip install tensorboard_logger
pip install wandb
pip install tsplib95