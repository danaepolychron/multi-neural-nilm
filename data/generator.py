from __future__ import print_function, division
import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import numpy as np
import torch.utils.data as data

    
class Seq2Seq(data.Dataset):
    
    """
    Defines a Dataset where each sample is a sequence of aggregate data and 
    the label is the corresponding sequence of the appliance meters data. 

    ...

    Attributes
    ----------
    mains : pandas.Series
        series containing the aggregate meter data  
    meters : pandas.DataFrame
        dataframe containing the appliance meters data 
    length : int
        the length of the sequence 
    speed : int
        the value used to limit the number of samples and minimise fitting time. Defaults to 1. (i.e no speed)
        
    train : boolean
    
    samples : int
        the number of samples in the dataset
    
    """
    
    
    def __init__(self, mains=None, meters=None, length=100, speed=1, train=False):
        
        self.mains =  mains
        self.meters = meters
        
        self.length = length
        self.train = train
        self.speed = speed
 
        self.samples = len(self.mains) // self.length*self.speed #to minimise fit time
             
        
    def __getitem__(self, index):
        
        # i increases by the sequence length
        i = index * self.length
        
        if self.train:      
            i = np.random.randint(0, len(self.mains) - self.length)
        
        X = self.mains.iloc[i:i+self.length].values.astype('float32')
        y = self.meters.iloc[i:i+self.length].values.astype('float32')

        X -= X.mean()
        
        return X, y

    def __len__(self):
        
        return self.samples 
    

class Seq2Point(data.Dataset):
    
    """
    Defines a Dataset where each sample is a sequence of aggregate data 
    and the label is the first point of the corresponding sequence of appliance meters data. 

    ...

    Attributes
    ----------
    mains : pandas.Series
        series containing the aggregate meter data  
    meters : pandas.DataFrame
        dataframe containing the appliance meters data 
    length : int
        the length of the sequence 
    speed : int
        the value used to limit the number of samples and minimise fitting time. Defaults to 1. (i.e no speed)
        
    train : boolean
    
    samples : int
        the number of samples in the dataset
    
    """
    
    
    def __init__(self, mains=None, meters=None, length=100, speed=1, train=False):
        
        self.mains =  mains
        self.meters = meters
        
        self.length = length
        self.train = train
        self.speed = speed
        
        self.samples = len(self.mains) // self.speed  #to minimise fit time
     
    
    def __getitem__(self, index):
        
        # i increases by one
        i = index
        
        if self.train:      
            i = np.random.randint(0, len(self.mains) - self.length)
        
        X = self.mains.iloc[i:i+self.length].values.astype('float32')
        y = self.meters.iloc[i].values.astype('float32')

        X -= X.mean()
        
        return X, y

    def __len__(self):
        
        return self.samples 
    
    
    