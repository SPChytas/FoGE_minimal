
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt 

from dataset import GraphDatasetSimple
from utils.logger import log, set_log
from utils.metrics import Accuracy, MSE



class OneHiddenModel(nn.Module):

    def __init__(self, input_dim=512, num_output=1, activation='sigmoid'):
        super(OneHiddenModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, 32)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(32, num_output)
        
        if (activation == 'sigmoid'):
            self.output_activation = nn.Sigmoid()
        elif (activation == 'softmax'):
            self.output_activation = nn.Softmax()
        elif (activation == 'identity'):
            self.output_activation = nn.Identity()
        else:
            raise ValueError('OneHiddenModel: unknown activation %s' %(activation))

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.output_activation(x)
        return x


set_log(2)




target = 'links_count'



model = OneHiddenModel(512, activation='sigmoid' if target=='has_cycle' else 'identity')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

if (target == 'has_cycle'):
    loss_function = nn.BCELoss()
else:
    loss_function = nn.MSELoss()

# train_dataset = GraphDatasetSimple('data/hypergraphqa/tasks/%s.csv' %(target), 'train', '1024.hrr')
# valid_dataset = GraphDatasetSimple('data/hypergraphqa/tasks/%s.csv' %(target), 'valid', '1024.hrr')
# test_dataset = GraphDatasetSimple('data/hypergraphqa/tasks/%s.csv' %(target), 'test', '1024.hrr')


data = GraphDatasetSimple('data/graphqa/tasks/%s.csv' %(target), '', '512.hrr', single_file=True)
train_dataset, valid_dataset, test_dataset, _ = random_split(data, [0.16, 0.02, 0.02, 0.8])


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)




for epoch in range(2000):

    pbar = tqdm(train_loader, total=len(train_loader))
    for X, y in pbar:

        y_pred = torch.squeeze(model(X))
        loss = loss_function(y_pred, y.to(torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description('Loss: %.8f' %(loss.item()))

    
    if (target == 'has_cycle'):
        metric = Accuracy()
    else:
        metric = MSE()
    metric.reset()

    pbar = tqdm(valid_loader, total=len(valid_loader))
    for X, y in pbar:
        
        y_pred = torch.round(torch.squeeze(model(X)))

        metric.update(y_pred.detach().numpy(), y.detach().numpy())
        res = metric.result()
        

        pbar.set_description('Accuracy: %.3f' %(res))



if (target == 'has_cycle'):
    metric = Accuracy()
else:
    metric = MSE()
metric.reset()




total_y = []
total_y_pred = []


pbar = tqdm(train_loader, total=len(train_loader))
for X, y in pbar:

    y_pred = torch.round(torch.squeeze(model(X)))

    metric.update(y_pred.detach().numpy(), y.detach().numpy())
    res = metric.result()

    total_y.extend(list(y.detach().numpy()))
    total_y_pred.extend(list(y_pred.detach().numpy()))

    pbar.set_description('Accuracy: %.3f' %(res))


plt.scatter(total_y, total_y_pred, alpha=0.01)
plt.savefig('train.png')
plt.close()






total_y = []
total_y_pred = []

metric.reset()

pbar = tqdm(test_loader, total=len(test_loader))
for X, y in pbar:

    y_pred = torch.round(torch.squeeze(model(X)))

    # for b in range(y_pred.shape[0]):
    #     print (y_pred[b], y[b])

    metric.update(y_pred.detach().numpy(), y.detach().numpy())
    res = metric.result()

    total_y.extend(list(y.detach().numpy()))
    total_y_pred.extend(list(y_pred.detach().numpy()))

    pbar.set_description('Accuracy: %.3f' %(res))


plt.scatter(total_y, total_y_pred, alpha=0.01)
plt.savefig('test.png')
plt.close()