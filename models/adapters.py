import torch
import torch.nn as nn 


class MLP(nn.Module):

    def __init__(self, input_dim=512, hidden_dims=[1024], output_dim=512, activation='identity'):
        super(MLP, self).__init__()

        self.all_dims = [input_dim] + hidden_dims + [output_dim]

        if (activation == 'sigmoid'):
            output_activation = nn.Sigmoid()
        elif (activation == 'softmax'):
            output_activation = nn.Softmax()
        elif (activation == 'identity'):
            output_activation = nn.Identity()
        else:
            raise ValueError('MLP: unknown activation %s' %(activation))

        self.layers = []
        for i in range(len(self.all_dims)-2):
            self.layers.append(nn.Linear(self.all_dims[i], self.all_dims[i+1])) 
            self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(self.all_dims[-2], self.all_dims[-1]))
        self.layers.append(output_activation)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x







