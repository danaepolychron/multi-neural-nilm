#need to review
from __future__ import print_function, division
import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import torch.utils.data as data




    
class Seq2Seq(data.Dataset):
    
    def __init__(self, data=None, length=100, train=False):
        
        super().__init__()
        
        self.mains = data['site meter'] / 2000.
        self.activations = data.drop(['site meter'], axis=1)
        self.length = length
        self.train = train
        self.samples = len(self.mains) // self.length
        
    def __getitem__(self, index):
        
        i = index * self.length
        
        if self.train:      
            i = np.random.randint(0, len(self.mains) - self.length)
        
        X = self.mains.iloc[i:i+self.length].values.astype('float32')
        y = self.activations.iloc[i:i+self.length].values.astype('float32')

        X -= X.mean()
        
        return X, y

    def __len__(self):
        
        return self.samples 
    
    
    def load(self, batch_size, shuffle):
        
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)
        
        
    

class Seq2Point(data.Dataset):
    
    def __init__(self, data=None, length=100, train=False):
        
        super().__init__()
             
        self.mains = data['site meter'] / 2000.
        self.activations = data.drop(['site meter'], axis=1)
        
        self.length = length
        self.train = train


        self.samples = len(self.mains) - self.length #needs change
        
    def __getitem__(self, index):
        
        i = index
        
        if self.train:      
            i = np.random.randint(0, len(self.mains) - self.length)
        
        X = self.mains.iloc[i:i+self.length].values.astype('float32')
        y = self.activations.iloc[i].values.astype('float32')

        X -= X.mean()
        
        return X, y

    def __len__(self):
        
        return self.samples 
    
    def load(self, batch_size, shuffle):
        
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)