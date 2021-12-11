# graph structure that allows transformations and stores all relevant information
# alternatively define operators that directly act on edge and node features
# a graph can be represented by a tuple (node_features, edge_features) each matrix might be sparse. 
# Multiple tensors of each kind can be stacked together to form a graph stack
class Graph:
    def __init__(self) -> None:
       self.node_features = None
       self.edge_features = None
       self.num_nodes = None
       self.num_edges = None
       self.num_edge_features = None
       self.num_node_features = None
    
    def transform(self):
        pass

    def sparsify(self):
        pass
