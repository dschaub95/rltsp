import torch
import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
from itertools import combinations, product
import math
import random

def get_group_travel_distances_sampling(selected_node_list, data, sampling_steps):
    """
    Calculates the travel distance for any number of samples tours (also 1)

    Args:
        selected_node_list ([type]): [description]
        data ([type]): [description]
        sampling_steps ([type]): [description]

    Returns:
        [type]: [description]
    """
    batch_s = selected_node_list.size(0)
    group_s = selected_node_list.size(1)
    tsp_size = selected_node_list.size(2)
    gathering_index = selected_node_list.unsqueeze(3).expand(batch_s, -1, tsp_size, 2)
    # shape = (batch, group, tsp_size, 2)
    # select only the original problems based on number of sampling steps
    orig_data = data[::sampling_steps, None, :, :].expand(-1, sampling_steps, tsp_size, 2).clone()
    # reshape data
    orig_data = torch.reshape(orig_data, (-1, tsp_size, 2))
    seq_expanded = orig_data[:, None, :, :].expand(batch_s, group_s, tsp_size, 2)

    ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
    # shape = (batch, group, tsp_size, 2)

    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
    # size = (batch, group, tsp_size)

    group_travel_distances = segment_lengths.sum(2)
    # size = (batch, group)
    return -1 * group_travel_distances

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape = (batch_s, problem, 2)
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape = (batch, problem, 1)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1-x, y), dim=2)
    dat3 = torch.cat((x, 1-y), dim=2)
    dat4 = torch.cat((1-x, 1-y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1-y, x), dim=2)
    dat7 = torch.cat((y, 1-x), dim=2)
    dat8 = torch.cat((1-y, 1-x), dim=2)

    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape = (8*batch, problem, 2)

    return data_augmented

class Transformation:
    """
    Transformation base class
    """
    def __init__(self, probability=0.5) -> None:
        self.invariant = None
        # probably built in random transformation sampling
        # keep track of previously applied transfos
        self.history = []
        # how likely is it that this transform gets applied
        self.probability = probability

    def reset(self):
        # clear history
        self.history = []

    def __call__(self, problem):
        raise NotImplementedError

class PomoTransformation(Transformation):
    pass

# should be implemented as stack of multiple transformation classes with call function and 
class TSPEucTransformer:
    
    def __init__(self) -> None:
        pass

    def load_TSP_from_nx(self, g):
        return np.array([g.nodes[k]['coord'] for k in g.nodes()])

    def save_TSP_as_nx(self, problem):
        edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(problem),2)]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        feature_dict = {k: {'coord': problem[k]} for k in range(problem.shape[0])} 
        nx.set_node_attributes(g, feature_dict)
        return g

    def scale_TSP(self, problem, scaler=1.0):
        if scaler == 1.0:
            return problem
        problem = self.center_TSP(problem, refit=False)
        problem = problem * scaler
        return self.translate_TSP(problem)

    def flip_TSP_coordinates(self, problem):
        return problem[:,::-1]

    def pomo_transform(self, problem, variant=0):
        assert problem.shape[1] == 2
        x = problem[:,0:1]
        y = problem[:,1::]
        if variant == 0:
            return np.concatenate((x, y), -1)
        elif variant == 1:
            return np.concatenate((1 - x, y), -1)
        elif variant == 2:
            return np.concatenate((x, 1 - y), -1)
        elif variant == 3:
            return np.concatenate((1 - x, 1 - y), -1)
        elif variant == 4:
            return np.concatenate((y, x), -1)
        elif variant == 5:
            return np.concatenate((1 - y, x), -1)
        elif variant == 6:
            return np.concatenate((y, 1 - x), -1)
        elif variant == 7:
            return np.concatenate((1 - y, 1 - x), -1)
    
    def reflect_TSP(self, problem, axis=0):
        # apply reflection matrix
        pass

    def flip_TSP_simple(self, problem, refit=False):
        # can be realized as a (N+1)D rotation along various axis
        dimension = problem.shape[1]
        assert dimension == 2
        problem = self.center_TSP(problem, refit)
        temp = np.concatenate((problem, np.zeros((problem.shape[0], 1))), -1)
        temp = rotate_3D(temp, degree=180)
        problem = temp[:,0:dimension]
        if refit:
            problem = self.fit_TSP_into_square(problem)
        return problem

    def flip_TSP(self, problem, flip_axis=0, refit=False):
        if flip_axis == 0:
            pass
        elif flip_axis == 1:
            problem =  self.flip_TSP_simple(problem)
        elif flip_axis == 2:
            problem = self.rotate_TSP(problem, degree=90, refit=refit)
            problem = self.flip_TSP_simple(problem)
            problem = self.rotate_TSP(problem, degree=-90, refit=refit)
        elif flip_axis == 3:
            problem = self.rotate_TSP(problem, degree=45, refit=refit)
            problem = self.flip_TSP_simple(problem)
            problem = self.rotate_TSP(problem, degree=-45, refit=refit)
        elif flip_axis == 4:
            problem = self.rotate_TSP(problem, degree=-45, refit=refit)
            problem = self.flip_TSP_simple(problem)
            problem = self.rotate_TSP(problem, degree=45, refit=refit)
        return problem
    
    def apply_PCA_to_TSP(self, problem, variant=1):
        dimension = problem.shape[1]
        if variant == 0:
            return problem
        pca = PCA(n_components=dimension) # center & rotate coordinates
        problem = pca.fit_transform(problem)
        return self.fit_TSP_into_square(problem)

    def rotate_TSP(self, problem, degree, refit=True):
        dimension = problem.shape[1]
        assert dimension == 2
        if degree == 0:
            return problem
        problem = self.center_TSP(problem, refit)
        problem = rotate_2D(problem, degree)
        if refit:
            return self.fit_TSP_into_square(problem)
        else:
            return self.translate_TSP(problem, shift=(0.5,0.5))

    def translate_TSP(self, problem, shift=(0.5,0.5)):
        dimension = problem.shape[1]
        assert dimension == 2
        x = problem[:,0:1]
        y = problem[:,1::]
        return np.concatenate((x - shift[0], y - shift[1]), -1)

    def fit_TSP_into_square(self, problem):
        dimension = problem.shape[1]
        maxima = []
        minima = []
        for k in range(dimension):
            maxima.append(np.max(problem[:,k]))
            minima.append(np.min(problem[:,k]))

        differences = [maxima[k] - minima[k] for k in range(dimension)]

        scaler = 1 / np.max(differences)

        for k in range(dimension):
            problem[:,k] = scaler * (problem[:,k] - minima[k])
        return problem

    def center_TSP(self, problem, refit=False):
        if refit:
            problem = self.fit_TSP_into_square(problem)
        return problem - 0.5

    def apply_random_transfo(self, problem, no_transfo=False):
        # keep track of applied transformations in a class variable
        # we can also retireve the number of transformations --> first 8 should always be pomo transformations
        if no_transfo:
            return problem
        # variants = [0, 1, 2, 3, 4, 5, 6, 7]
        variants = [1, 2, 3, 4, 5, 6, 7]
        # flip_axis = [0, 1, 2, 3, 4]
        flip_axis = [1, 2, 3, 4]
        # degrees = np.arange(0, 360, 45)
        degrees = np.arange(45, 360, 45)
        pcas = [0, 1]
        # make random choices
        axis = random.choice(flip_axis)
        degree = random.choice(degrees)
        variant = random.choice(variants)
        pca = random.choice(pcas)

        problem = self.pomo_transform(problem, variant=variant)
        # problem = self.apply_PCA_to_TSP(problem, pca)
        problem = self.flip_TSP(problem, axis)
        problem = self.rotate_TSP(problem, degree, refit=True)
        return problem


# potentially later implement each transform in a seperate class with default / invariant value etc
class RandomTSPEucTransformation(TSPEucTransformer):
    """
    Realizes random scaling, reflection, rotation, translation, mirroring
    """
    def __init__(self, pomo=True, flipping=True, scaling=True, rotation=True, translation=True, pca=False,  pomo_first=False) -> None:
        super().__init__()
        self.applied_transfos = []
        # if true then the first seven transformations are all pomo transformations
        self.pomo_first = pomo_first
        if pomo:
            self.pomo_variants = np.arange(1,8)
        else:
            self.pomo_variants = [0]
        if flipping:
            self.flip_axis = np.arange(1, 5)
        else:
            self.flip_axis = [0]
        if rotation:
            self.degrees = np.linspace(15, 345, 11) # 30 degrees
        else:
            self.degrees = [0.0]
        if scaling:
            self.scalers = np.linspace(0.75, 1.05, 10)
        else:
            self.scalers = [1.0]
        if translation:
            self.translation_vectors = list(product(np.linspace(-0.1, 0.1, 3), np.linspace(-0.1, 0.1, 3)))
        else:
            self.translation_vectors = [(0.0, 0.0)]
        if pca:
            self.pca = [0, 1]
        else:
            self.pca = [0]
        # self.possible_transfos = product(self.pomo_variants, self.flip_axis, self.degrees, self.scalers)
    
    def reset(self):
        self.applied_transfos = []

    @property
    def num_applied_transfos(self):
        return len(self.applied_transfos)
    
    def check_if_transfo_available(self, transfo):
        # maybe check if transfo is just similar (ignoring the scaling)
        if transfo in self.applied_transfos:
            return False
        else:
            return True

    def nested_transform(self, problem, transfo_id):
        problem = self.pomo_transform(problem, variant=transfo_id[0])
        problem = self.flip_TSP(problem, transfo_id[1])
        problem = self.rotate_TSP(problem, transfo_id[2], refit=True)
        problem = self.scale_TSP(problem, scaler=transfo_id[3])
        problem = self.apply_PCA_to_TSP(problem, variant=transfo_id[4])
        return problem

    def sample_random_transfo(self):
        # iterate until a new transfo is found (other option is to apply random choice diretly to a list of ids)
        # this is more efficient if the number of possible ids is large
        while True:
            # generate random transfo_id
            pomo_variant = random.choice(self.pomo_variants)
            flip_axis = random.choice(self.flip_axis)
            degree = random.choice(self.degrees)
            scaler = random.choice(self.scalers)
            pca = random.choice(self.pca)
            transfo_id = [pomo_variant, flip_axis, degree, scaler, pca]
            # check that transfo has not been applied yet
            if self.check_if_transfo_available(transfo_id):
                break
        return transfo_id

    def __call__(self, problem):
        if self.pomo_first and self.num_applied_transfos < 7:
            pomo_variant = self.num_applied_transfos + 1
            transfo_id = [pomo_variant] + [0, 0.0, 1.0, 0] # invariant values for each transform
        else:
            transfo_id = self.sample_random_transfo()
        # apply transfo
        transformed_problem = self.nested_transform(problem, transfo_id)
        # add transfo to memory
        self.applied_transfos.append(transfo_id)
        return transformed_problem



def rotate_3D(vectors, degree):
    assert vectors.shape[-1] == 3
    radians = (degree / 360) * 2 * math.pi
    rotmatrix = np.array([[1, 0, 0],
                          [0, math.cos(radians), -math.sin(radians)],
                          [0, math.sin(radians), math.cos(radians)]])

    return vectors @ rotmatrix

def rotate_2D(vectors, degree):
    assert vectors.shape[-1] == 2
    radians = (degree / 360) * 2 * math.pi
    rotmatrix = np.array([[math.cos(radians),-math.sin(radians)],
                          [math.sin(radians),math.cos(radians)]])
    return vectors @ rotmatrix
    
