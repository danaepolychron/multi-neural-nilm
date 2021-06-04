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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from sklearn.model_selection import KFold

from data import Environment, get_on_off
from data.generator import Seq2Seq, Seq2Point 

from models import FCN, ResNet, ConvGRU, ConvLSTM, FCN_AE
from experiments.metrics import validation_report, evaluation_report, confusion_matrix_report, roc_report

from utils.path_finder import SOURCES, PRETRAINED, RESULTS
from utils.constants import MAINS, APPLIANCES



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experiment(object): 
    
    """
    Defines all the necessary utility functions to conduct an experiment. 

    ...

    Attributes
    ----------
    scenario : int
        the scenario that defines the environment params
    appliances : dict 
        the target appliances with their corresponding min-off duration, min-on duration, power threshold
    model : inherits from torch.nn.Module
        the model used for the training and inference phases of the experiment
    window : int
        the sequence length of each sample in the dataset
    epochs : int
        the number of epochs used to train the model
    lrn_rate : int or float
        the learning rate of the optimiser
    batch_size : int
        the number of samples per gradient update
    speed : int
        the value used to limit the number of samples and minimise fitting time. Defaults to 1 (i.e no speed)
    
    """
    
    
    def __init__(self, scenario=1):
        
        super().__init__()
        
        self.scenario = scenario 
        self.appliances = {}
        self.model = None
        self.window = None
        self.epochs = None
        self.lrn_rate = None
        self.batch_size = None
        self.speed = 1
    
    
    def setup_environment(self, mode):
        
        """Sets the environment as defined in the scenario configuration file."""

        env = Environment(scenario = self.scenario, mode=mode)
        env.configure_environment()

        filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, mode))
        env.save_environment(env.generate_environment(), filename)
            
        
    def load_environments(self):
        
        """Loads the training and testing environments.
    
        Returns
        -------
        tuple of pandas.DataFrame, each one corresponds to the training and testing set. 

        """
        datasets = {}
        
        for mode in ["train", "test"]:
            
            filename = os.path.join(SOURCES,"scenario-{}-{}-set.feather".format(self.scenario, mode))
            if not os.path.exists(filename):
                self.setup_environment(mode=mode)
        
            datasets[mode] = pd.read_feather(filename)
            datasets[mode].set_index('index', inplace=True)
        
        
        return datasets["train"], datasets["test"]
    
    
    def get_appliances(self):
        
        """Returns the target appliances from the specified scenario. 
        
        Returns
        -------
        dict,  the target appliances with their corresponding min-off duration, min-on duration, power threshold.
        
        """
        
        env = Environment(scenario = self.scenario, mode="train")
        env.configure_environment()
        
        return env.get_appliances()
      
        
    def setup_running_params(self, model, window=100, epochs=10, lrn_rate=0.0001, batch_size=32, speed=1): 
        
        """Sets the running params for the experiment."""
        
        self.model = model 
        self.window = window
        self.epochs = epochs
        self.lrn_rate = lrn_rate
        self.batch_size = batch_size
        self.speed = speed
        self.appliances = self.get_appliances()
        
        
    
    def train(self, train_generator, valid_generator, filename):
        
        """Trains the model on the specified target appliances.
        
        Parameters
        ----------
        train_generator : torch.utils.data.DataLoader, iterable with the training samples and corresponding labels
        valid_generator : torch.utils.data.DataLoader, iterable with the validation samples and corresponding labels
        filename : str, the path where the model's checkpoints are saved

        Returns
        -------
        tuple of Lists[floats],  the training and validation losses at successive epochs.  
        
        """
        
        train_losses = []
        valid_losses = [] 
        
        avg_train_losses = []
        avg_valid_losses = []

        min_loss = np.inf

        optimizer = optim.Adam(self.model.parameters(), lr=self.lrn_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(1, self.epochs + 1):

            self.model.train() 
            for batch, (X, y) in enumerate(train_generator, 1):

                X = X.unsqueeze(1).to(device)
                y = y.to(device)

                optimizer.zero_grad()
                
                if self.model.is_autoencoder(): 
                    output = self.model(X).permute(0,2,1)
                else: 
                    output = self.model(X)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())


            self.model.eval()
            
            for X, y in valid_generator:
                
                X = X.unsqueeze(1).to(device)
                y = y.to(device)
                
                if self.model.is_autoencoder(): 
                    output = self.model(X).permute(0,2,1)
                else:
                    output = self.model(X)
       
                loss = criterion(output, y)
                valid_losses.append(loss.item())
            

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            
            
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))
            epoch_msg = (f'[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f} ')

            print(epoch_msg)

            train_losses = []
            valid_losses = []

            if valid_loss < min_loss:
                print(f'Validation loss decreased ({min_loss:.6f} --> {valid_loss:.6f}).')

                min_loss = valid_loss
                
                
        torch.save(self.model.state_dict(), filename)

        return  avg_train_losses, avg_valid_losses
        
    
    def test(self, generator, appliance, filename):
        
        """Evaluates the model on the specified target appliances.
        
        Parameters
        ----------
        generator : torch.utils.data.DataLoader, iterable with the testing samples and corresponding labels
        appliance : int, appliance index
        filename : str, the path where the model's checkpoints are saved

        Returns
        -------
        tuple of numpy.arrays, the ground truth and the the predicted result
        
        """
        
        y_true = []
        predictions = []
        
        self.model.load_state_dict(torch.load(filename))

        self.model.eval()
        with torch.no_grad():
            for X, y in generator:
                
                X = X.unsqueeze(1).to(device)
                
                if self.model.is_autoencoder(): 
                    y = y.permute(0,2,1)[:, appliance]
                else:
                    y = y[:, appliance]

                output = self.model(X)
                output = torch.sigmoid(output[:,appliance])

                
                y_true.append(y.contiguous().view(-1).detach().cpu().numpy())
                
                predictions.append(output.contiguous().view(-1).detach().cpu().numpy())

        y_true = np.hstack(y_true)
        predictions = np.hstack(predictions)

        return y_true, predictions
    
        
    def reset_weights(self, m):
    
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            
            
    def crossvalidate(self, dataset, n_splits=5):
        
        """Applies cross validation.
        
        Parameters
        ----------
        dataset : torch.utils.data.DataSet, contains the testing samples and corresponding labels
        n_splits : int, number of folds. Must be at least 2.

        Returns
        -------
        dict, result scores per each target appliance and per model, i.e macro and micro scores
        
        """
        
        results = {}
    
        # Set fixed random number seed
        torch.manual_seed(42)
        kfold = KFold(n_splits=n_splits, shuffle=False)
        
        directory = os.path.join(PRETRAINED, "{}".format(self.model.name))
        if not os.path.exists(directory):
            os.makedirs(directory)

        for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
            
            print("Training {} for fold {} ...".format(self.model.name, fold+1))


            train_generator = DataLoader(dataset, 
                                         batch_size=self.batch_size, 
                                         shuffle=False, 
                                         sampler=SubsetRandomSampler(train_index))
            test_generator = DataLoader(dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False,
                                        sampler=SubsetRandomSampler(test_index))

            # reset model weights
            self.model.apply(self.reset_weights)
          

            filename = os.path.join(directory, "scenario-{}-{}-min-checkpoint-{}.pth".format(self.scenario, self.window, fold))
            train_loss, test_loss = self.train(train_generator, test_generator, filename)
  
            
            print("Saving evaluation results for fold {} ...".format(fold+1))
            
            predictions = pd.DataFrame(columns=self.appliances)
            y_test = pd.DataFrame(columns=self.appliances)

            for i, (appliance, thresholds) in enumerate(self.appliances.items()):

                y_true, y_predicted = self.test(test_generator, i, filename)
                y_test[appliance] = y_true
                
                # predictions[appliance] = get_on_off(y_predicted, 0.5, thresholds[0], thresholds[1])
                predictions[appliance] = y_predicted.round()

            
            target_results = evaluation_report(y_test.values, predictions.values, targets=self.appliances, mode="target")
            model_results = evaluation_report(y_test.values, predictions.values, mode="averaged")

            results[fold] = {"target_results" : target_results, "model_results": model_results}
            self.save_experiment_results(results[fold]["target_results"], results[fold]["model_results"], fold=fold) 
            
        
        print("Done!")
        max_score = -np.inf
        for fold in results.keys():
            
            if results[fold]["model_results"].loc["macro","F1 Score"] >= max_score:
                
                max_score = results[fold]["model_results"].loc["macro","F1 Score"]
                best_results = results[fold]
                
        return best_results 
        
    
    
    def save_experiment_results(self, target_results, model_results, fold=None):
        
        """Saves experiment results.
        
        Parameters
        ----------
        target_results : pandas.DataFrame, contains the scores per target appliance
        model_results : pandas.DataFrame, contains the macro and micro scores 
        fold : int, the fold index
        
        """
        
        directory = os.path.join(RESULTS, "scenario-{}/{}".format(self.scenario,self.model.name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if fold is None:             
            target_path = "{}-min-target-results.csv".format(self.window)
            model_path = "{}-min-model-results.csv".format(self.window)
        else: 
            target_path = "{}-min-target-results-{}.csv".format(self.window, fold+1)
            model_path = "{}-min-model-results-{}.csv".format(self.window, fold+1)
            
        target_filename = os.path.join(directory, target_path)
        target_results.to_csv(target_filename)
        
        model_filename = os.path.join(directory, model_path)
        model_results.to_csv(model_filename)
        
    
    def run(self, cv=2): 
        
           
        train, test = self.load_environments()
        
        
        if self.model.is_autoencoder():
            
            ds_train = Seq2Seq(mains=train[MAINS], meters=train[APPLIANCES], length=self.window, speed=self.speed, train=True)
            ds_test = Seq2Seq(mains=test[MAINS], meters=test[APPLIANCES], length=self.window, speed=self.speed, train=False)
        else: 
            ds_train = Seq2Point(mains=train[MAINS], meters=train[APPLIANCES], length=self.window, speed=self.speed, train=True)
            ds_test = Seq2Point(mains=test[MAINS], meters=test[APPLIANCES], length=self.window, speed=self.speed, train=False)
        
      
        ds_merged = ConcatDataset([ds_train, ds_test])
        
        results = self.crossvalidate(dataset=ds_merged, n_splits=cv)
        
        self.save_experiment_results(results["target_results"], results["model_results"]) 
        
            
            
                                    
                                    
                                    
                                    
                                    
                                   


                                    

    
