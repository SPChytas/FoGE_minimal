


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
from tqdm.auto import tqdm
import pickle

from utils.logger import log




class LLMDataset(Dataset):

    def __init__(self, labels_file, split, method, tokenizer, preload=True, num_graph_embs=1, single_file=False):

        self.labels_file = labels_file
        self.task = labels_file.split(os.path.sep)[-1][:-4] 
        self.split = split
        self.method = method
        self.preload = preload

        self.num_graph_embs = num_graph_embs

        labels_pd = pd.read_csv(os.path.join(self.labels_file))
        self.labels_pd = labels_pd[labels_pd['path'].str.contains(self.split)]

        log('LLMDataset: keep %d rows for split %s' %(self.labels_pd.shape[0], self.split), 'info')

        self.data = None

        if (single_file):
            embs = pickle.load(open(os.path.join(os.path.sep.join(self.labels_file.split(os.path.sep)[:-2]), 'graphs', 'all_embs.%s' %(self.method)), 'rb'))
        

        if (self.preload):

            log('LLMDataset: preloading symbols', 'info')

            self.data = {'symbol': [], 
                         'input': [], 
                         'output': []}
            
            for i, row in tqdm(self.labels_pd.iterrows(), total=self.labels_pd.shape[0]):

                if (i > 1000000):
                    break

                question_tok = tokenizer(row['question'],
                                         add_special_tokens=True,
                                         truncation=True,
                                         max_length=1024,
                                         return_tensors='pt')['input_ids']
                
                answer_tok = tokenizer(str(row['answer']) + '\n',
                                       add_special_tokens=False,
                                       truncation=True,
                                       max_length=1024,
                                       return_tensors='pt')['input_ids']

                input_tok = torch.cat((question_tok, answer_tok[..., :-1]), dim=-1)
                output_tok = torch.cat((torch.full(question_tok.shape, -1), answer_tok), dim=-1)[..., 1:]

                self.data['input'].append(input_tok)
                self.data['output'].append(output_tok)

                if (self.num_graph_embs == 1):
                    if (single_file):
                        self.data['symbol'].append(torch.unsqueeze(embs[row['path'].split(os.path.sep)[-1]], 0))
                    else:
                        self.data['symbol'].append(torch.unsqueeze(torch.load(row['path'] + '.%s' %(self.method)), 0))
                else:
                    self.data['symbol'].append(torch.unsqueeze(torch.load(row['path'] + '.%s.nodes' %(self.method)), 0))

    def __len__(self):
        return len(self.data['symbol'])
        # return self.labels_pd.shape[0]
    
    def __getitem__(self, idx):
        if (self.preload):
            return self.data['symbol'][idx], self.data['input'][idx], self.data['output'][idx]
        

    def collate_fn(self, batch):

        symbols = [x for x, _, _ in batch]
        symbols = torch.cat(symbols, 0)
        
        inputs = [x for _, x, _ in batch]
        outputs = [x for _, _, x in batch]

        input_lens = [x.shape[-1] for x in inputs]
        max_len = np.max(input_lens)# + self.num_graph_embs

        inputs = [torch.cat((torch.zeros((1, max_len - input_lens[i]), dtype=torch.int64), inputs[i]), dim=-1) for i in range(len(batch))]
        outputs = [torch.cat((torch.full((1, max_len - input_lens[i]), -1, dtype=torch.int64), outputs[i]), dim=-1) for i in range(len(batch))]

        inputs = torch.cat(inputs, 0)
        outputs = torch.cat(outputs, 0)

        return symbols, inputs, outputs, torch.LongTensor(input_lens)




class GraphDatasetSimple(Dataset):

    def __init__(self, labels_file, split, method, single_file=False):

        self.labels_file = labels_file
        self.task = labels_file.split(os.path.sep)[-1][:-4] 
        self.split = split
        self.method = method


        labels_pd = pd.read_csv(os.path.join(self.labels_file))
        self.labels_pd = labels_pd[labels_pd['path'].str.contains(self.split)]

        log('GraphDataset: keep %d rows for split %s' %(self.labels_pd.shape[0], self.split), 'info')

        self.data = None

        log('GraphDataset: preloading symbols', 'info')

        self.data = {'symbol': [], 
                     'output': []}
        
        if (single_file):
            embs = pickle.load(open(os.path.join(os.path.sep.join(self.labels_file.split(os.path.sep)[:-2]), 'graphs', 'all_embs.%s' %(self.method)), 'rb'))
        
        for i, row in tqdm(self.labels_pd.iterrows(), total=self.labels_pd.shape[0]):

            if (self.task == 'has_cycle'):
                self.data['output'].append(0 if row['answer']=='No' else 1)
                if (single_file):
                    self.data['symbol'].append(torch.unsqueeze(embs[row['path'].split(os.path.sep)[-1]], 0))
                else:
                    self.data['symbol'].append(torch.unsqueeze(torch.load(row['path'] + '.%s' %(self.method)), 0))
            else:
                self.data['output'].append(int(row['answer']))
                if (single_file):
                    self.data['symbol'].append(torch.unsqueeze(embs[row['path'].split(os.path.sep)[-1]], 0))
                else:
                    self.data['symbol'].append(torch.unsqueeze(torch.load(row['path'] + '.%s' %(self.method)), 0))
                
                
    def __len__(self):
        return self.labels_pd.shape[0]
    
    def __getitem__(self, idx):
        return self.data['symbol'][idx], self.data['output'][idx]
        

   

