import os
import numpy as np 
import torch 
from transformers import AutoTokenizer, AutoModel

from utils.attr_info import CategoricalAttr, MultiCategoricalAttr, VectorAttr
from utils.logger import log



def _generate_orthonormal_vectors(n_vectors, dim):

    # orthogonal
    if (n_vectors <= dim):
        log('utils._generate_orthonormal_vectors: getting exactly orthonormal vectors (n_vectors %d <= dim %d)' %(n_vectors, dim), 'info')

        A = np.random.normal(size=(dim, dim))
        Q = np.linalg.qr(A)[0][:n_vectors, :]
    # almost orthogonal    
    else:
        log('utils._generate_orthonormal_vectors: getting approximately orthonormal vectors (n_vectors %d > dim %d)' %(n_vectors, dim), 'info')

        Q = np.random.normal(size=(n_vectors, dim))
        norms = np.linalg.norm(Q, axis=1)
        Q = Q/norms.reshape(-1, 1)

        cs = np.abs(Q @ Q.T) - np.eye(n_vectors)
        log('utils._generate_orthonormal_vectors: maximum (abs) cosine similarity: %.2f' %(np.max(cs)), 'debug')
    
    return Q





def random_vectors(attr_list, dim):

    num_vectors = sum([attr.len() for attr in attr_list])
    total_vectors = _generate_orthonormal_vectors(num_vectors, dim)

    start = 0
    for attr in attr_list:
        if ((not isinstance(attr, VectorAttr)) and attr.len() > 0):
            attr.vectors = torch.Tensor(total_vectors[start: start + attr.len()])
            start += attr.len()
        elif (isinstance(attr, VectorAttr)):
            attr.map = torch.FloatTensor(np.random.normal(scale=1/max(attr.v_dim, dim), size=(attr.v_dim, dim)))



def pretrained_vectors(attr_list, vectors_path, dim):

    vectors_list = np.load(os.path.join(vectors_path, '%d.npy' %(dim)), allow_pickle=True)

    # getting precalculated vectors/maps/transformers
    vectors_dict = {}
    for attr in vectors_list:
        if (isinstance(attr, VectorAttr)):
            vectors_dict[attr.name] = (attr.map, attr.transformer)
        else:
            vectors_dict[attr.name] = (attr.vectors, attr.transformer)
        vectors_dict[attr.name].requires_grad = False

    # assigning precalculated values to attributes list
    remain_vectors = 0 
    for attr in attr_list:
        if (attr.name in vectors_dict):
            if (isinstance(attr, VectorAttr)):
                attr.map, attr.transformer = vectors_dict[attr.name]
            else:
                attr.vectors, attr.transformer = vectors_dict[attr.name] 
        else:
            remain_vectors += attr.len()

    # generate randomly any remaining attributes
    if (remain_vectors > 0):
        log('load_pretrained_vectors: %d vectors were not found. generating them randomly...' %(remain_vectors), 'info')

        total_vectors = _generate_orthonormal_vectors(remain_vectors, dim)

        start = 0
        for attr in attr_list:
            if (attr.name in vectors_dict):
                continue
            if ((not isinstance(attr, VectorAttr)) and attr.len() > 0):
                attr.vectors = torch.Tensor(total_vectors[start: start + attr.len()])
                start += attr.len()
            elif (isinstance(attr, VectorAttr)):
                attr.map = torch.FloatTensor(np.random.normal(scale=1/max(attr.v_dim, dim), size=(attr.v_dim, dim)))






class TextEncoder:

    def __init__(self, model):

        if (model == 'roberta'):
            self.tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', 
                                                    do_lower_case=False,
                                                    do_basic_tokenize=True,
                                                    use_fast=True)
            self.text_encoder = AutoModel.from_pretrained('FacebookAI/roberta-base')
            self.text_encoder.eval()
        else:
            raise ValueError('text_vectors: Unknown model %s' %(model))
        
    def encode(self, text):
        tokens = self.tokenizer(str(text), return_tensors="pt", max_length=512)
        response = self.text_encoder(**tokens)[0]
        response = torch.flatten(response.mean(1).detach().cpu())
        response /= torch.linalg.norm(response)
        return response


def text_vectors(attr_list, model):

    model = TextEncoder(model)

    for attr in attr_list:
        if (isinstance(attr, CategoricalAttr) or isinstance(attr, MultiCategoricalAttr)):
            values = []
            for v in attr.values:
                response = model.encode(v)
                values.append(response)
            attr.vectors = torch.vstack(values)
        else:
            attr.text_encoder = model
