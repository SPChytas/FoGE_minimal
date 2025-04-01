from abc import ABC, abstractmethod
import numpy as np 
import torch 

from utils.logger import log



class Attr(ABC):

    @abstractmethod
    def __init__(self):
        self.vectors = None 
        self.map = None 
        self.text_encoder = None
        self.transformer = lambda x: x

    @abstractmethod
    def len(self):
        pass 

    @abstractmethod
    def get_id(self, x):
        pass 

    @abstractmethod
    def get_vector(self, x):
        pass


class CategoricalAttr(Attr):
    '''
    Categorical attributes
    ----------------------
    In this case the set of all possible values is finite
    We assume one vector per distinct value
    '''

    def __init__(self, values, name=''):
        super().__init__()

        self.name = name
        
        self.values = sorted(values)
        self.values_ids = {self.values[i]: i for i in range(len(self.values))}  

    def len(self):
        return len(self.values)
    
    def get_id(self, x):
        if (x not in self.values_ids.keys()):
            log('CategoricalAttr %s: value not found. Return default' %(self.name), 'debug')
            
        return self.values_ids.get(x, 0)

    def get_vector(self, x):
        assert type(self.vectors) != type(None), 'CategoricalAttr: Attribute\'s vectors not initialized'

        return self.transformer(self.vectors[self.get_id(x)])



class MultiCategoricalAttr(Attr):
    '''
    MultiCategorical attributes
    ----------------------
    In this case the set of all possible values is finite
    We assume one vector per distinct value

    In contrast to CategoricalAttr, each input can be a collection of multiple values
    - we assume x is a string of all the multiple values, separated by '+'
    '''

    def __init__(self, values, name=''):
        super().__init__()

        self.name = name
        
        self.values = sorted(values)
        self.values_ids = {self.values[i]: i for i in range(len(self.values))}  

    def len(self):
        return len(self.values)
    
    def get_id(self, x):
        if (x not in self.values_ids.keys()):
            log('MultiCategoricalAttr %s: value not found. Return default' %(self.name), 'debug')
            
        return self.values_ids.get(x, 0)

    def get_vector(self, x):
        assert type(self.vectors) != type(None), 'MultiCategoricalAttr: Attribute\'s vectors not initialized'
        
        values = x.split('+')
        return [self.transformer(self.vectors[self.get_id(x)]) for x in values]



class VectorAttr(Attr):
    '''
    VectorAttr
    ----------
    here each value is a vector of a specific dimensionality (1 or more)
    and the number of possible vectors is infinite

    to preserve the relationship between the vectors in the original space
    we use a linear transformation to map the vectors to the new space
    (after we first normalize the vectors, to avoid any explosions in the binding)
    '''

    def __init__(self, means, stds, name=''):
        super().__init__()
        
        self.name = name
        
        self.means = torch.FloatTensor(list(means)).flatten()
        self.stds = torch.FloatTensor(list(stds)).flatten()
        assert self.means.shape == self.stds.shape, 'means and stds list are of different shape'

        self.v_dim = len(self.means)

        
    def len(self):
        return 0
    
    def get_id(self, x):
        return None
    
    def get_vector(self, x):
        assert type(self.map) != type(None), 'VectorAttr: Attribute\'s mapping not initialized' 
        
        if (type(x) != str):
            x = str(x)
        return self.transformer((torch.Tensor(np.fromstring(x, sep=' ')) - self.means)/self.stds @ self.map)
        


class TextAttr(Attr):
    '''
    TextAttr
    --------
    In this case, the mapping of the values into vectors is given by a pretrained text model 

    TODO: normalization of output vectors? right now normalized in text_encoder.encode()
    '''

    def __init__(self, name=''):
        super().__init__()

        self.name = name

    def len(self):
        return 0
    
    def get_id(self, x):
        return None
    
    def get_vector(self, x):
        assert type(self.text_encoder) != type(None) 
        return self.text_encoder.encode(x)