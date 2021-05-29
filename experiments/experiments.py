#need to review
from __future__ import print_function, division
import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from data import Environment, get_status
from data.generator import Seq2Seq, Seq2Point 

from models import FCN, ResNet, ConvGRU, ConvLSTM, FCN_AE
from experiments.metrics import validation_report, evaluation_report, confusion_matrix_report, roc_report

from utils.path_finder import NILMTK_RAW, SCENARIOS, SOURCES, PRETRAINED


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experiment(object): 
    
    """
    A class used to represent an experiment 

    Attributes
    ----------
    atr : str
        a formatted string to print out what the animal says

    Methods
    -------
    method(sound=None)
        Prints the animals name and what sound it makes
    """
    
    def __init__(self, scenario=1):
        
        super().__init__()
        
        self.scenario = scenario 
        self.appliances = []
        self.model = None
        self.window = None
        self.epochs = None
        self.lrn_rate = None
        self.batch_size = None
        self.repeat = 1
        
        self.results_checkpoint = None
    
    
    
    def setup_environments(self):

        for mode in ['train', 'validation', 'test']: 
            
            env = Environment(scenario = self.scenario, mode=mode)
            env.configure_environment()

            filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, mode))
            env.save_environment(env.generate_environment(), filename)
            
        
    def load_environments(self):
        
            
        train_filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, "train"))
        train = pd.read_feather(train_filename)
        train.set_index('index', inplace=True)
        
        validation_filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, "validation"))
        validation = pd.read_feather(validation_filename)
        validation.set_index('index', inplace=True)
        
        test_filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, "test"))
        test = pd.read_feather(test_filename)
        test.set_index('index', inplace=True)
        
        return train, validation,test
    
    
    def setup_appliances(self):
        
        env = Environment(scenario = self.scenario, mode="train")
        env.configure_environment()
        
        return env.get_appliances()
      
        
    def setup_running_params(self, model, window=100, epochs=10, lrn_rate=0.0001, batch_size=32, repeat=1): 
        
        
        self.model = model 
        self.window = window
        self.epochs = epochs
        self.lrn_rate = lrn_rate
        self.batch_size = batch_size
        self.repeat = 1
        
        self.appliances = self.setup_appliances()
        
        
    
    def train(self, train_loader, valid_loader, test_loader, filename):
        
        train_losses = []
        valid_losses = []
        test_losses = []
        avg_train_losses = []
        avg_valid_losses = [] 
        avg_test_losses = [] 

        min_loss = np.inf

        optimizer = optim.Adam(self.model.parameters(), lr=self.lrn_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(1, self.epochs + 1):

            self.model.train() 
            for batch, (data, target) in enumerate(valid_loader, 1):

                data = data.unsqueeze(1).to(device)
                target = target.to(device)

                optimizer.zero_grad()
                
                if self.model.is_autoencoder(): 
                    output = self.model(data).permute(0,2,1)
                else: 
                    output = self.model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())


            self.model.eval()
            for data, target in valid_loader:
                data = data.unsqueeze(1).to(device)
                target = target.to(device)
                
                if self.model.is_autoencoder(): 
                    output = self.model(data).permute(0,2,1)
                else:
                    output = self.model(data)
       
                loss = criterion(output, target)
                valid_losses.append(loss.item())
            
            self.model.eval()
            for data, target in test_loader:
                data = data.unsqueeze(1).to(device)
                target = target.to(device)
                
                if self.model.is_autoencoder(): 
                    output = self.model(data).permute(0,2,1)
                else:
                    output = self.model(data)
       
                loss = criterion(output, target)
                test_losses.append(loss.item())


            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            test_loss = np.average(test_losses)
            
            
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            avg_test_losses.append(test_loss)

            epoch_len = len(str(self.epochs))
            epoch_msg = (f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f} ' +
                         f'test_loss: {test_loss:.5f} ')

            print(epoch_msg)

            train_losses = []
            valid_losses = []
            test_losses = []

            if valid_loss < min_loss:
                print(f'Validation loss decreased ({min_loss:.6f} --> {valid_loss:.6f}).  Saving model ...')
                torch.save(self.model.state_dict(), filename)
                min_loss = valid_loss

        self.model.load_state_dict(torch.load(filename))

        return  avg_train_losses, avg_valid_losses, avg_test_losses
        
    
    def test(self, loader, appliance):
        
        X_true = []
        y_true = []
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                
                X = X.unsqueeze(1).to(device)
                
                if self.model.is_autoencoder(): 
                    y = y.permute(0,2,1)[:, appliance]
                else:
                    y = y[:, appliance]

                output = self.model(X)
                output = torch.sigmoid(output[:,appliance])

                
                #X_true.append(data[:,:,0].contiguous().view(-1).detach().cpu().numpy())
                y_true.append(y.contiguous().view(-1).detach().cpu().numpy())
                
                predictions.append(output.contiguous().view(-1).detach().cpu().numpy())

        #X_true = np.hstack(X_true)
        y_true = np.hstack(y_true)
        predictions = np.hstack(predictions)

        return y_true, predictions
        
    
    def save_experiment(self, target_results, model_results):
        
        target_filename = os.path.join(RESULTS, "{}/target/scenario-{}.csv".format(self.model.name,self.scenario))
        target_results.to_csv(target_filename, index=False)
        
        model_filename = os.path.join(RESULTS, "{}/model/scenario-{}.csv".format(self.model.name,self.scenario))
        model_results.to_csv(model_filename, index=False)
        
        
            
        

        
    
    def run(self): 
        
        if not os.listdir(SOURCES):
            self.setup_environments()
           
        train, test = self.load_environments()
        
        if self.model.is_autoencoder():
            
            ds_train = Seq2Seq(data=train, length=self.window, train=True)
            ds_test = Seq2Seq(data=test, length=self.window, train=False)
        else:          
            ds_train = Seq2Point(data=train, length=self.window, train=True)
            ds_test = Seq2Point(data=test, length=self.window, train=False)
        

        dl_train = DataLoader(dataset = ds_train, batch_size = self.batch_size, shuffle=True)
        dl_valid = DataLoader(dataset = ds_test, batch_size = self.batch_size, shuffle=False)
        dl_test = DataLoader(dataset = ds_test, batch_size = self.batch_size, shuffle=False)
        
        target_results = []
        model_results = []
        
        for i in range(self.repeat):
            
            filename = os.path.join(PRETRAINED, "{}/scenario-{}-run-{}.pth".format(self.model.name, self.scenario, i))
            train_loss, valid_loss, test_loss = self.train(dl_train, dl_valid, dl_test, filename)
            predictions = pd.DataFrame(columns=self.appliances)
            y_test = pd.DataFrame(columns=self.appliances)
            
            for i, (appliance, thresholds) in enumerate(self.appliances.items()):
                
                X_true, y_true, y_predicted = self.test(dl_test, i)
                y_test[appliance] = y_true
                predictions[appliance] = get_status(y_predicted, 0.5, thresholds[0], thresholds[1])
            
            
            target_results.append(evaluation_report(y_test.values, predictions.values, targets=self.appliances, mode="target"))
            model_results.append(evaluation_report(y_test.values, predictions.values, mode="averaged"))
            
            
        target_results_avg = np.average(target_results)
        model_results_avg = np.average(model_results)
        
        self.save_experiment(target_results_avg, model_results_avg)   
            
            
                                    
                                    
                                    
                                    
                                    
                                   


                                    

    
