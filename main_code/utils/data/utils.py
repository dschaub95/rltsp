import networkx as nx
import numpy as np
import random
import torch
import os
import json
from itertools import combinations

# convert node coords into nx graph
def convert_tsp_to_nx(node_coords):
    num_nodes = node_coords.shape[0]
    edges = [
        (s[0], t[0], np.linalg.norm(s[1] - t[1]))
        for s, t in combinations(enumerate(node_coords), 2)
    ]
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    feature_dict = {k: {"coord": node_coords[k]} for k in range(num_nodes)}
    nx.set_node_attributes(g, feature_dict)
    return g


def calc_tour_length(node_feats, sol):
    # calculate length of the tour
    differences = node_feats[sol, :] - np.roll(node_feats[sol, :], shift=-1, axis=0)
    summed_squares = np.sum(np.square(differences), axis=1)
    length = np.sum(np.sqrt(summed_squares))
    return length


def save_tsp():
    # generic code for saving tsp
    # add to data utils
    pass


def sample_and_save_subset(dataset, save_path, sample_size=128, seed=37):
    random.seed(37)
    indices = random.sample(range(len(dataset)), sample_size)
    subset = torch.utils.data.Subset(dataset, indices)
    # save data in new folder
    # make sure length stays the same
    for idx, data in enumerate(subset):
        idx_str = f"{idx}".zfill(len(str(len(subset))))
        problem_name = f"tsp_{idx_str}"
        instance_path = f"{save_path}/{problem_name}"
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)
        node_feats = data[0]
        np.savetxt(f"{instance_path}/node_feats.txt", node_feats)
        sol = data[2]
        length = calc_tour_length(node_feats, sol)
        assert data[1] == length
        # save solution data to folder
        solution_data = {
            "problem_name": problem_name,
            "opt_tour_length": length,
            "opt_tour": sol.tolist(),
        }
        with open(f"{instance_path}/solution.json", "w") as f:
            # indent=2 is not needed but makes the file human-readable
            json.dump(solution_data, f, indent=2)
