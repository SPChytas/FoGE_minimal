
import numpy as np 
import torch
import networkx as nx

from utils.vectors_creators import random_vectors, text_vectors, TextEncoder
from utils.vsa import HRR
from utils.attr_info import CategoricalAttr
from utils.logger import log




class GraphEncoder:

    def __init__(self, attr_list, method='hrr', dim=512, vectors='random', nodewise=False, levels=1):
        '''
        Args
        ----
        attr_list: [attr_info.Attr]
            a list of node attributes. It must contain at least the following attribute: CategoricalAttr(range(max_nodes), 'node_id')
        method: str (default: 'hrr')
            how to merge the vectors
        dim: int (default: 512)
            the vectors dimensionality. Used only if vectors = 'random'
        vectors: str (default: 'random')
            how to set the corresponding vector of each attribute
            if 'random' the vectors are chosen to be (almost) orthogonal
            else the vectors are generated using the <vectors> text encoder
        nodewise: bool (default: False)
            if True, generate one embedding per node
            if False, generate one embedding for the whole graph
        levels: int (default: 1)
            encoding depth 
        '''

        log('GraphEncoder: encode edge attributes (currently ignored)', 'todo')


        self.levels = levels

        # Check attr_list 
        self.node_id_attr_id = -1
        for i, attr in enumerate(attr_list):
            if (attr.name == 'node_id'):
                self.node_id_attr_id = i 
        if (self.node_id_attr_id == -1):
            raise AttributeError('Attribute node_id is missing')


        self.attr_list =attr_list
        self.attr_names = sorted([attr.name for attr in self.attr_list])
        self.attr_list.append(CategoricalAttr(['number of nodes', 'number of edges'], 'size'))

        if (method == 'hrr'):
            self.method = HRR()
        else:
            raise ValueError('Unknown method %s' %(method))
        
        self.dim = dim
        self.vectors_method = vectors 
        self.nodewise = nodewise

       
        if (self.vectors_method == 'random'):
            random_vectors(self.attr_list, self.dim) 
        else:
            text_vectors(self.attr_list, self.vectors_method)
            

        if (self.vectors_method == 'random'):
            self.signature = '%s_%d_%s_%s_%d' %('*'.join(self.attr_names), self.dim, method, 'nodewise' if self.nodewise else 'single', self.levels)
        else:
            self.signature = '%s_%s_%s_%s_%d' %('*'.join(self.attr_names), self.vectors_method, method, 'nodewise' if self.nodewise else 'single', self.levels)
        log ('GraphEncoder: initialized with signature %s' %(self.signature), 'info')

    def encode(self, graph):

        adjacency_matrix = nx.to_numpy_array(graph)
        num_nodes = adjacency_matrix.shape[0]


        prev_node_representations = [self.attr_list[self.node_id_attr_id].get_vector(i) for i in range(num_nodes)]

        for _ in range(self.levels):

            # Encode the edges
            node_representations = []
            for i in range(num_nodes):
                node_representation = torch.zeros(size=prev_node_representations[i].shape)
                for j in range(num_nodes):
                    if (adjacency_matrix[i, j] != 0):
                        node_representation = self.method.bundle(node_representation, 
                                                                self.method.bind(prev_node_representations[i], 
                                                                                 prev_node_representations[j]))
                node_representations.append(node_representation)    

            # Encode all node attributes
            for attr in self.attr_list:
                if (attr.name == 'node_id' or attr.name == 'size'):
                    continue

                node_attrs = nx.get_node_attributes(graph, attr.name)
                node_attrs = {int(k): v for k, v in node_attrs.items()}

                for i in node_attrs.keys():
                    
                    node_representations[i] = self.method.bundle(node_representations[i],
                                                                self.method.bind(prev_node_representations[i], 
                                                                                 attr.get_vector(node_attrs[i])))

            # Encode all edge attributes
            for attr in self.attr_list:

                edge_attrs = nx.get_edge_attributes(graph, attr.name)
                edge_attrs = {(int(k1), int(k2)): v for (k1, k2), v in node_attrs.items()}

                for i, j in edge_attrs.keys():

                    node_representations[i] = self.method.bundle(node_representations[i],
                                                                self.method.bind(self.method.bind(prev_node_representations[i], 
                                                                                                  prev_node_representations[j]),
                                                                                 attr.get_vector(node_attrs[i])))

            prev_node_representations = [v/torch.linalg.norm(v) for v in node_representations]





        vector_representation = torch.vstack(node_representations)
        # Reduce to a single vector if not nodewise
        if (not self.nodewise):
            single_representation = vector_representation[0]
            for i in range(1, vector_representation.shape[0]):
                single_representation = self.method.bundle(single_representation, vector_representation[i])
            vector_representation = self.method.bundle(single_representation, 
                                                       self.method.bind(self.attr_list[-1].get_vector('number of nodes'), 
                                                                        node_representations[num_nodes-1]))

        return vector_representation 
        
    def get_vectors(self):
        return self.attr_list

    










