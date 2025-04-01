from abc import ABC, abstractmethod

from math import sqrt
import torch 
from torch.fft import fft, ifft


class VSA(ABC):

    @abstractmethod
    def bind(self, x, y):
        pass 

    @abstractmethod
    def bundle(self, x, y):
        pass 

    @abstractmethod
    def unbind(self, x, y):
        pass


class HRR(VSA):

    def bind(self, x, y):
        return torch.real(ifft(torch.multiply(fft(x), fft(y))))

    def bundle(self, x, y):
        return x + y 
    
    def unbind(self, x, y):
        x_inv = torch.flip(x, dims=[-1])
        return self.bind(torch.roll(x_inv, 1, dims=-1), y)
    

class MAP(VSA):

    def bind(self, x, y):
        return x * y
    
    def bundle(self, x, y):
        return x + y 
    
    def unbind(self, x, y):
        x_inv = 1/(x + 1e-6)
        return self.bind(x_inv, y)


class VTB(VSA):

    def _get_binding_matrix(self, x):
        
        dim = len(x)
        dim_prime = int(sqrt(x))

        assert dim_prime*dim_prime == dim, 'VTB bind: dimension %d is not a perfect square' %(dim)

        return sqrt(dim_prime)*torch.kron(torch.eye(dim_prime), x.reshape((dim_prime, dim_prime)))

    def _get_unbinding_matrix(self, x):
        return self._get_binding_matrix(x).T

    def bind(self, x, y):
        x = x.flatten()
        y = y.flatten()

        return self._get_binding_matrix(x) @ y 
    
    def bundle(self, x, y):
        return x + y 
    
    def unbind(self, x, y):
        x = x.flatten()
        y = y.flatten()

        return self._get_unbinding_matrix(x) @ y 