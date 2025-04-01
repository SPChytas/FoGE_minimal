
import pandas as pd
import os
from tqdm.auto import tqdm
import networkx as nx
import numpy as np
import torch 
import copy
import datasets
import pickle
import dgl 
import argparse

from joblib import Parallel, delayed

import task_generators.graph_tasks as graph_tasks
import task_generators.hypergraph_tasks as hypergraph_tasks
import task_generators.protein_tasks as protein_tasks

from utils.attr_info import CategoricalAttr, TextAttr, VectorAttr
from utils.encoders import GraphEncoder

from utils.logger import log, set_log

########################################### <LLM prompt/response> ###########################################
'''
Create the tasks that accompany each dataset. The task are basic graph understanding tasks, such as
number of nodes, has cycle, etc.
These tasks will then be used for the training and evaluation of the FoGE capabilities

Args
----
graphs_path: str
    the parent folder of all the .graphml files
output_path: str
    the parent folder in which we will store all the generated .csv files
'''

def create_tasks_graphqa(graphs_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    tasks = {'node_count': graph_tasks.node_count,
             'node_degree': graph_tasks.node_degree,
             'edge_count': graph_tasks.edge_count,
             'edge_existence': graph_tasks.edge_existence,
             'has_cycle': graph_tasks.cycle,
             'num_triangles': graph_tasks.triangles}
    
    for task_desc, task_fun in tasks.items():

        dataset = []

        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]
        for file_name in tqdm(files, total=len(files), desc=task_desc):

            if (file_name[-7:] != 'graphml'):
                continue
            
            G = nx.read_graphml(file_name)
            G_path = file_name[:-8]

            dataset.extend(task_fun(G, G_path))

        df = pd.DataFrame(dataset)
        for c in df.columns:
            df[c] = df[c].astype(str)
        df.to_csv(os.path.join(output_path, '%s.csv' %(task_desc)), index=False)


def create_tasks_graphreasoning(graphs_path):

    tasks = {'sc': graph_tasks.substructure_count,
             'mts': graph_tasks.maximum_triplet_sum,
             'mts_named': graph_tasks.maximum_triplet_sum_named,
             'sp': graph_tasks.shortest_path,
             'bgm': graph_tasks.bipartite_graph_matching}
    
    for task_desc, task_fun in tasks.items():

        dataset = []

        path = os.path.join(graphs_path, task_desc.split('_')[0])
        os.makedirs(os.path.join(path, 'task'), exist_ok=True)

        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path)) for f in fn]
        for file_name in tqdm(files, total=len(files), desc=task_desc):

            if (file_name[-7:] != 'graphml'):
                continue
            
            G = nx.read_graphml(file_name)
            G_path = file_name[:-8]

            dataset.extend(task_fun(G, G_path))

        df = pd.DataFrame(dataset)
        for c in df.columns:
            df[c] = df[c].astype(str)
        df.to_csv(os.path.join(path, 'task', '%s.csv' %(task_desc)), index=False)


def create_tasks_hypergraphqa(graphs_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    tasks = {'node_count': hypergraph_tasks.node_count,
             'node_degree': hypergraph_tasks.node_degree,
             'edge_count': hypergraph_tasks.edge_count,
             'edge_existence': hypergraph_tasks.edge_existence}
    
    for task_desc, task_fun in tasks.items():

        dataset = []

        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]
        for file_name in tqdm(files, total=len(files), desc=task_desc):

            if (file_name[-12:] != 'hnx_dict.pkl'):
                continue

            with open(file_name, 'rb') as f:
                G = pickle.load(f)
            
            G_path = file_name[:-13]

            dataset.extend(task_fun(G, G_path))

        df = pd.DataFrame(dataset)
        for c in df.columns:
            df[c] = df[c].astype(str)
        df.to_csv(os.path.join(output_path, '%s.csv' %(task_desc)), index=False)


def create_tasks_jaffe(graphs_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    tasks = {'amino_acid_count': protein_tasks.node_count,
             'amino_acid_type': protein_tasks.node_type,
             'links_count': protein_tasks.edge_count}
    
    for task_desc, task_fun in tasks.items():

        dataset = []

        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]
        for file_name in tqdm(files, total=len(files), desc=task_desc):

            if (file_name[-7:] != 'graphml'):
                continue
            
            G = nx.read_graphml(file_name)
            G_path = file_name[:-8]

            dataset.extend(task_fun(G, G_path))

        df = pd.DataFrame(dataset)
        for c in df.columns:
            df[c] = df[c].astype(str)
        df.to_csv(os.path.join(output_path, '%s.csv' %(task_desc)), index=False)
########################################### </LLM prompt/response> ###########################################




########################################### <Dataset encoding> ###########################################
'''
Convert a collection of .graphml files to embeddings, using the corresponding encoder. 
The conversion is dataset-specific, since each dataset is of a different graph form, with different node attributes

Args
----
graphs_path: str
    the parent folder of all the .graphml files
method: str (default: 'hrr')
    how to merge the vectors
dim: int (default: 512)
    the vectors dimensionality. Used only if vectors = 'random'
vectors: str (default: 'random')
    how to set the corresponding vector of each attribute
    if 'random' the vectors are chosen to be (almost) orthogonal
    else the vectors are generated using the <vectors> text encoder
bins: int (default: 10)
    used only if there are continuous attributes
    how many distinct bins to create for each continuous attribute
nodewise: bool (default: False)
    if True, generate one embedding per node
    if False, generate one embedding for the whole graph
'''

def encode_graphqa(graphs_path, method='hrr', dim=512, vectors='random', nodewise=False, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## GraphQA has no node attributes, so here we just create the node_id attribute
    max_nodes = 0
    for file_name in tqdm(files, total=len(files), desc='encode_graphqa: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())


    encoder = GraphEncoder([CategoricalAttr(list(range(max_nodes)), 'node_id')],
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)

    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_graphqa: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))


def encode_graphreasoning(graphs_path, method='hrr', dim=512, vectors='random', nodewise=False, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## GraphReasoning has up to four extra attributes: 1. desc, 2. name, 3. value, 4. symbol
    max_nodes = 0
    values_dict = {'name': set(),
                   'value': set(),
                   'symbol': set()}

    for file_name in tqdm(files, total=len(files), desc='encode_graphreasoning: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())

        for attr in values_dict:
            values_dict[attr].union(set(list(nx.get_node_attributes(G, attr).values())))

    if (vectors == 'random'):
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in values_dict.items() if len(attr_values) > 1]
    else:
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in values_dict.items() if len(attr_values) > 1] + \
                    [TextAttr('desc')]
        
    encoder = GraphEncoder(attr_list,
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)

    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_graphreasoning: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))


def encode_jaffe(graphs_path, method='hrr', dim=512, vectors='random', bins=10, nodewise=False, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## GraphReasoning has four extra attributes: 1. type, 2-4. coords
    max_nodes = 0
    categorical_dict = {'type': set()}
    continuous_dict = {'x_coord': (float('inf'), -float('inf')),
                       'y_coord': (float('inf'), -float('inf')),
                       'z_coord': (float('inf'), -float('inf'))}

    for file_name in tqdm(files, total=len(files), desc='encode_jaffe: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())

        for attr in categorical_dict:
            categorical_dict[attr].union(set(list(nx.get_node_attributes(G, attr).values())))

        for attr in continuous_dict:
            values = list(nx.get_node_attributes(G, attr).values())
            continuous_dict[attr] = (min(continuous_dict[attr][0], min(values)), max(continuous_dict[attr][1], max(values)))

    if (vectors == 'random'):
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in categorical_dict.items() if len(attr_values) > 1] + \
                    [ContinuousAttr(attr_values[0], attr_values[1], bins, attr) for attr, attr_values in continuous_dict.items()]
    else:
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in categorical_dict.items() if len(attr_values) > 1] + \
                    [TextAttr(attr) for attr, attr_values in continuous_dict.items()]
        
    encoder = GraphEncoder(attr_list,
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)


    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_jaffe: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))


def encode_PPID(graphs_path, method='hrr', dim=512, vectors='random', nodewise=True, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## GraphReasoning has up to four extra attributes: 1. desc, 2. name, 3. value, 4. symbol
    max_nodes = 0
    feats_dim = None

    for file_name in tqdm(files, total=len(files), desc='encode_PPID: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())

        feats = list(nx.get_node_attributes(G, 'feats').values())
        feats = [np.fromstring(f, sep=' ') for f in feats]
        feats_dim = len(feats[0])

    attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                [VectorAttr(feats_dim, 'feats')]

    encoder = GraphEncoder(attr_list,
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)

    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_PPID: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))


def encode_obnb(graphs_path, method='hrr', dim=512, vectors='random', nodewise=True, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## obnb has up one extra attribute: node features
    max_nodes = 0
    feats_dim = None

    for file_name in tqdm(files, total=len(files), desc='encode_obnb: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())

        feats = list(nx.get_node_attributes(G, 'feats').values())
        feats = [np.fromstring(f, sep=' ') for f in feats]
        feats_dim = len(feats[0])

    attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                [VectorAttr(feats_dim, 'feats')]

    encoder = GraphEncoder(attr_list,
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)

    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_obnb: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))


def encode_GraphWiz(graphs_path, method='hrr', dim=512, vectors='random', nodewise=False, levels=1):

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(graphs_path)) for f in fn]

    # Part 1: get attributes
    ## GraphReasoning has up to four extra attributes: 1. desc, 2. name, 3. value, 4. symbol
    max_nodes = 0
    node_values_dict = {'node_weight': set()}
    edge_values_dict = {'edge_weight': set()}

    for file_name in tqdm(files, total=len(files), desc='encode_GraphWiz: get attributes statistics'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        max_nodes = max(max_nodes, G.number_of_nodes())

        for attr in node_values_dict:
            node_values_dict[attr] = node_values_dict[attr].union(set(list(nx.get_node_attributes(G, attr).values())))
        for attr in edge_values_dict:
            edge_values_dict[attr] = edge_values_dict[attr].union(set(list(nx.get_edge_attributes(G, attr).values())))

    print ((node_values_dict['node_weight']))
    print ((edge_values_dict['edge_weight']))

    if (vectors == 'random'):
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in node_values_dict.items() if len(attr_values) > 1] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in edge_values_dict.items() if len(attr_values) > 1] 
    else:
        attr_list = [CategoricalAttr(list(range(max_nodes)), 'node_id')] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in node_values_dict.items() if len(attr_values) > 1] + \
                    [CategoricalAttr(attr_values, attr) for attr, attr_values in edge_values_dict.items() if len(attr_values) > 1] + \
                    [TextAttr('desc')]
        
    print ('Max nodes: %d' %(max_nodes))

    encoder = GraphEncoder(attr_list,
                           method,
                           dim,
                           vectors,
                           nodewise,
                           levels)

    # Part 2: encode all graphs
    for file_name in tqdm(files, total=len(files), desc='encode_GraphWiz: encoding graphs'):
        if (file_name[-7:] != 'graphml'):
            continue
        
        G = nx.read_graphml(file_name)
        emb = encoder.encode(G)
        torch.save(emb, file_name[:-8] + '.%s' %(encoder.signature))
        
    torch.save(encoder.get_vectors(), os.path.join(graphs_path, 'vectors.%s' %(encoder.signature)))
########################################### </Dataset encoding> ###########################################




########################################### <Graph format> ###########################################
'''
Convert a collection of files to the standard graphml format. 
The conversion is dataset-specific, since each dataset is provided in a different format.
'''

def graphreasoning_to_graphs(graphs_path):
    '''
    First preprocessing step for the GraphReasoning dataset
    Brings them into the typical format: 1 graphml file per graph, splitted into 3 folders (train/val/test)
    '''

    task = graphs_path.split(os.path.sep)[-1]

    if (task in ['mts', 'sc']):
        num_nodes = 15
    elif (task in ['sp', 'bgm']):
        num_nodes = 20
    else:
        raise ValueError
    
    
    os.makedirs(os.path.join(graphs_path, 'graphs', 'train'), exist_ok=True)
    os.makedirs(os.path.join(graphs_path, 'graphs', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(graphs_path, 'graphs', 'test'), exist_ok=True)

    dataset = datasets.load_from_disk(graphs_path)
    split = pickle.load(open(os.path.join(graphs_path, '%s_split.pkl' %(task)), 'rb'))
    edge_index = pickle.load(open(os.path.join(graphs_path, '%s_edge.pkl' %(task)), 'rb'))

    num_graphs = len(split['train']) + len(split['valid']) + len(split['test'])

    edge_pos = 0

    for graph_id in tqdm(range(num_graphs)):

        g = nx.Graph()

        node_list = []
        for node_id in range(num_nodes*graph_id, num_nodes*(graph_id + 1)):
            node_list.append((node_id%num_nodes, 
                              {'desc': dataset[node_id]['desc'],
                               'name': dataset[node_id].get('name', '-'),
                               'value': dataset[node_id].get('value', '-'),
                               'symbol': dataset[node_id].get('symbol', '-'),
                               'label': dataset[node_id].get('label', '-')}))
        g.add_nodes_from(node_list)

        edge_list = []
        while (edge_pos < edge_index.shape[1] and num_nodes*graph_id <= edge_index[0, edge_pos] < num_nodes*(graph_id + 1)):
            edge_list.append((edge_index[0, edge_pos].item()%num_nodes, edge_index[1, edge_pos].item()%num_nodes))
            edge_pos += 1
        g.add_edges_from(edge_list)

        if (graph_id*num_nodes in split['train']):
            nx.write_graphml(g, os.path.join(graphs_path, 'graphs', 'train', '%d.graphml' %(graph_id)))
        elif (graph_id*num_nodes in split['valid']):
            nx.write_graphml(g, os.path.join(graphs_path, 'graphs', 'valid', '%d.graphml' %(graph_id)))
        elif (graph_id*num_nodes in split['test']):
            nx.write_graphml(g, os.path.join(graphs_path, 'graphs', 'test', '%d.graphml' %(graph_id)))
        else:
            raise KeyError


def jaffe_to_graphs(proteins_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(proteins_path)) for f in fn]
    for proteins_file in files:
        
        if (proteins_file.split('.')[-1] != 'pkl'):
            continue

        with open(proteins_file, 'rb') as f:
            proteins = pickle.load(f)

        for protein in tqdm(proteins, total=len(proteins), desc='converting proteins to nx.Graphs'):

            seq = list(protein.str)
            nodes_info = [(i, 
                           {'type': seq[i], 
                            'x_coord': protein.coords[i, 0].item(),
                            'y_coord': protein.coords[i, 1].item(),
                            'z_coord': protein.coords[i, 2].item()}) for i in range(len(seq))]
            # edges_info = [(protein.edge_index[0][i].item(), 
            #                protein.edge_index[1][i].item(), 
            #                {'weight': protein.edge_weight[i//2].item()}) for i in range(0, protein.edge_index.shape[1], 2)]
            edges_info = [(protein.edge_index[0][i].item(), 
                           protein.edge_index[1][i].item()) for i in range(protein.edge_index.shape[1])]
            
            g = nx.Graph()
            g.add_nodes_from(nodes_info)
            g.add_edges_from(edges_info)

            graph_name = '%s_%s' %(proteins_file.split(os.path.sep)[-1].split('.')[0], protein.name)

            nx.write_graphml(g, os.path.join(output_path, '%s.graphml' %(graph_name)))


def PPID_to_graphs(output_path):

    for split in ['train', 'valid', 'test']:

        os.makedirs(os.path.join(output_path, split, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

        data = dgl.data.PPIDataset(split)

        for i in range(len(data)):

            g = dgl.to_networkx(data[i])
            nx.set_node_attributes(g, {j: np.array2string(data[i].ndata['feat'][j].numpy())[1:-1] for j in range(g.number_of_nodes())}, 'feats')
            labels = data[i].ndata['label']
            
            nx.write_graphml(g, os.path.join(output_path, split, 'graphs', '%d.graphml' %(i)))
            np.save(os.path.join(output_path, split, 'labels', '%d' %(i)), labels)
            

def obnb_to_graphs(pickle_folder, output_path):

    for pickle_file in os.listdir(pickle_folder):

        if (pickle_file[-3:] != 'pkl'):
            continue

        current_output_path = os.path.join(output_path, pickle_file.split('.')[0])

        os.makedirs(os.path.join(current_output_path, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(current_output_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(current_output_path, 'split'), exist_ok=True)


        data = pickle.load(open(os.path.join(pickle_folder, pickle_file), 'rb'))
        nodes_info = [i for i in range(data.num_nodes)]
        edges_info = [(data.edge_index[0][i].item(), 
                       data.edge_index[1][i].item()) for i in range(data.edge_index.shape[1])]
        
        g = nx.Graph()
        g.add_nodes_from(nodes_info)
        g.add_edges_from(edges_info)
        nx.set_node_attributes(g, {j: np.array2string(data.x[j].numpy())[1:-1] for j in range(g.number_of_nodes())}, 'feats')

        nx.write_graphml(g, os.path.join(current_output_path, 'graphs', 'protein.graphml'))
        np.save(os.path.join(current_output_path, 'labels', 'labels'), data.y.numpy())
        np.save(os.path.join(current_output_path, 'split', 'train'), data.train_mask.numpy())
        np.save(os.path.join(current_output_path, 'split', 'val'), data.val_mask.numpy())
        np.save(os.path.join(current_output_path, 'split', 'test'), data.test_mask.numpy())


def phred50_to_graphs(proteins_path, output_path, n_jobs=10):

    def _convert_pkl_to_graphml(output_path, proteins_file):

        protein = pd.read_picle(proteins_file)

        seq = list(protein.sequence_H + protein.sequence_L)
        nodes_info = [(i, 
                        {'type': seq[i], 
                        'coords': protein.coords[i].item(),
                        'x_coord': protein.coords[i, 0].item(),
                        'y_coord': protein.coords[i, 1].item(),
                        'z_coord': protein.coords[i, 2].item()}) for i in range(len(seq))]
        # edges_info = [(protein.edge_index[0][i].item(), 
        #                protein.edge_index[1][i].item(), 
        #                {'weight': protein.edge_weight[i//2].item()}) for i in range(0, protein.edge_index.shape[1], 2)]
        edges_info = [(protein.edge_index[0][i].item(), 
                        protein.edge_index[1][i].item()) for i in range(protein.edge_index.shape[1])]
        
        g = nx.Graph()
        g.add_nodes_from(nodes_info)
        g.add_edges_from(edges_info)

        graph_name = '%s_%s' %(proteins_file.split(os.path.sep)[-1].split('.')[0], protein.name)

        nx.write_graphml(g, os.path.join(output_path, '%s.graphml' %(graph_name)))


    os.makedirs(output_path, exist_ok=True)

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(proteins_path)) for f in fn]
    files = [f for f in files if f.split('.')[-1] == 'pkl']
    
    log ('phred50_to_graphs: use %d threads' %(n_jobs), 'info')
    Parallel(n_jobs=n_jobs)(delayed(_convert_pkl_to_graphml)(output_path, f) for f in files)
########################################### </Graph format> ###########################################









if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='graphqa')
    parser.add_argument('--method', type=str, default='hrr')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--vectors', type=str, default='random')
    parser.add_argument('--bins', type=int, default=10)

    parser.add_argument('--nodewise', action='store_true')
    parser.add_argument('--levels', type=int, default=1)

    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    set_log(args.verbose)

    if (args.dataset == 'graphqa'):
        log('encoding graphQA graphs...', 'info')
        encode_graphqa('data/graphqa/graphs', args.method, args.dim, args.vectors, args.nodewise, args.levels)
        log('Done', 'info')

        log('creatting graphQA tasks...', 'info')
        create_tasks_graphqa('data/graphqa/graphs', 'data/graphqa/tasks')
        log('Done', 'info')

    elif (args.dataset == 'graphreasoning'):
        for task in ['bgm', 'mts', 'sc', 'sp']:
            graphreasoning_to_graphs(os.path.join('data/graphreasoning', task))
            encode_graphreasoning(os.path.join('data/graphreasoning', task), args.method, args.dim, args.vectors, args.nodewise, args.levels)
        create_tasks_graphreasoning('data/graphreasoning')
    
    elif (args.dataset == 'hypergraphqa'):
        print ('not implemented yet')
    
    elif (args.dataset == 'PPID'):
        log('converting PPID to graphml files...', 'info')
        PPID_to_graphs('data/PPID')
        log('Done', 'info')

        log('encoding PPID graphs...', 'info')
        encode_PPID('data/PPID/', args.method, args.dim, args.vectors, args.nodewise, args.levels)
        log('Done', 'info')

        log ('create_tasks_PPID not implemented yet', 'info')
    
    elif (args.dataset == 'jaffe'):
        jaffe_to_graphs('data/jaffe/jaffe_abbert', 'data/jaffe/graphs')
        encode_jaffe('data/jaffe/graphs', args.method, args.dim, args.vectors, args.nodewise, args.levels)
        create_tasks_jaffe('data/jaffe/graphs', 'data/jaffe/tasks')
    
    elif (args.dataset == 'obnb'):
        log('converting obnb to graphml files...', 'info')
        obnb_to_graphs('data/obnb/obnbench_data', 'data/obnb/')
        log('Done', 'info')

        log('encoding obnb graphs...', 'info')
        for name in ['BioGRID_DISEASES', 'ComPPIHumanInt_DISEASES', 'FunCoup_DISEASES', 'HumanBaseTopGlobal_DISEASES', 'HuMAP_DISEASES',
                     'OmniPath_DISEASES', 'ProteomeHD_DISEASES', 'STRING_DISEASES', 'BioGRID_DisGeNET', 'ComPPIHumanInt_DisGeNET', 'FunCoup_DisGeNET',
                     'HumanBaseTopGlobal_DisGeNET', 'HuMAP_DisGeNET', 'OmniPath_DisGeNET', 'ProteomeHD_DisGeNET', 'STRING_DisGeNET', 'BioGRID_GOBP',
                     'ComPPIHumanInt_GOBP', 'FunCoup_GOBP', 'HumanBaseTopGlobal_GOBP', 'HuMAP_GOBP', 'OmniPath_GOBP', 'ProteomeHD_GOBP', 'STRING_GOBP',
                     'BioPlex_DISEASES', 'ConsensusPathDB_DISEASES', 'HIPPIE_DISEASES', 'HumanNet_DISEASES', 'HuRI_DISEASES', 'PCNet_DISEASES',
                     'SIGNOR_DISEASES', 'BioPlex_DisGeNET', 'ConsensusPathDB_DisGeNET', 'HIPPIE_DisGeNET', 'HumanNet_DisGeNET', 'HuRI_DisGeNET',
                     'PCNet_DisGeNET', 'SIGNOR_DisGeNET', 'BioPlex_GOBP', 'ConsensusPathDB_GOBP', 'HIPPIE_GOBP', 'HumanNet_GOBP', 'HuRI_GOBP',
                     'PCNet_GOBP', 'SIGNOR_GOBP']:
            encode_obnb(os.path.join('data/obnb/', name, 'graphs'), args.method, args.dim, args.vectors, args.nodewise, args.levels)
        log('Done', 'info')
    

    elif (args.dataset == 'GraphWiz'):

        log('encoding GraphWiz graphs...', 'info')
        encode_GraphWiz('data/GraphWiz/graphs', args.method, args.dim, args.vectors, args.nodewise, args.levels)
        log('Done', 'info')