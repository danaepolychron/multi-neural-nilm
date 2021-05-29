#need to review
from __future__ import print_function, division
import warnings

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd
import numpy as np

from nilmtk import DataSet, MeterGroup
from nilmtk.electric import get_activations
from nilmtk.timeframe import TimeFrame

from utils.path_finder import NILMTK_RAW, SCENARIOS
 

class Environment(object): 
    
    def __init__(self, scenario=1, mode='train'):
        
        super().__init__()
        
        self.scenario = scenario
        self.mode = mode
        
        self.source = None
        self.dataset = None
        self.buildings = None
        self.sample_period = None  
        self.metergroups = []
        self.appliances = []
        
    def configure_environment(self): 
        
        filename = os.path.join(SCENARIOS,"scenario_{}.json".format(self.scenario))
        with open(filename) as file:
            conf = json.load(file)
        
        self.source = os.path.join(NILMTK_RAW,conf['{}_source'.format(self.mode)])
        self.dataset = DataSet(self.source).metadata['name']
        self.buildings = conf['{}_buildings'.format(self.mode)] 
        self.sample_period = conf['sample_period']
        
        self.metergroups = self.configure_metergroups(conf, self.mode)
        self.appliances = self.configure_appliances(conf)
            
        
    def configure_metergroups(self, conf, mode):
        
        windows = conf['{}_windows'.format(mode)]   
        
        metergroups = [None]*len(self.buildings)
        
        for i, building in enumerate(self.buildings) :
            
            dataset = DataSet(self.source)
            dataset.set_window(start= windows[i][0], end=windows[i][1])
            metergroups[i]= dataset.buildings[building].elec
            
        return metergroups
    

    def configure_appliances(self, conf):
        
        for appliance, thresholds in conf['targets'].items():
            
            for single_building in self.metergroups :
        
                meter = single_building[appliance]
                if not meter:
                    raise NameError('There is no such meter in MeterGroup')

                meter.min_off_duration = thresholds[0] #min_off duration
                meter.min_on_duration = thresholds[1] #min_on duration
                meter.on_power_threshold = thresholds[2] #power_threshold  
                
        return conf['targets']
            
    
    def get_appliances(self):
        return self.appliances
            
        
    def get_activations_chunk(self, metergroups, appliance, chunk_section):
        
        meter = metergroups[appliance]
        chunk = next(meter.power_series(sections=[chunk_section]))
        
        activations = get_activations(chunk = chunk, 
                                      min_off_duration= meter.min_off_duration,     
                                      min_on_duration=meter.min_on_duration,
                                      on_power_threshold=meter.on_power_threshold,
                                      border=0)  
        
        return activations
        
    
    def align_meters(self, metergroups, appliance, chunk, chunk_section): 
    
        activations_chunk = self.get_activations_chunk(metergroups, appliance, chunk_section)

        chunk[appliance] = 0
        
        if activations_chunk:  
            for i in range(len(activations_chunk)):
                mask = (chunk.index >= activations_chunk[i].index[0]) & (chunk.index <= activations_chunk[i].index[-1])
                chunk.loc[mask, appliance] = 1

        return chunk[appliance] 
    
    def generate_single_building(self, building, i):
        
        print("Preparing {} building {} ...".format(self.dataset, self.buildings[i])) 
        
        mains = building.mains()
        good_sections = mains.good_sections()
        mains = mains.power_series(sample_period=self.sample_period, sections=good_sections)

        sections = []
        while True:

            try:
                chunk = next(mains)
            except StopIteration:
                break

            if len(chunk) < 2:
                break

            chunk = pd.DataFrame({'site meter': chunk})
            chunk_timeframe = TimeFrame(chunk.index[0], chunk.index[-1])

            for appliance in self.appliances: 
                chunk[appliance] = self.align_meters(building, appliance, chunk, chunk_timeframe)
            
            sections.append(chunk)

        if sections:    
            return pd.concat(sections)
      
        else:
            return pd.DataFrame(columns=["site meter"].append(self.appliances))
        
    
    def generate_multiple_building(self):

        buildings = [None] * len(self.metergroups)
        
        for i, single_building in enumerate(self.metergroups):
            
            buildings[i] = self.generate_single_building(self, single_building, i)
            
        
        if any(k is not None for k in buildings): 
            
            data = pd.concat(buildings)
            return data
        
        else:
            return pd.DataFrame(columns=["site meter"].append(self.appliances))
            

    
    def generate_environment(self):
        
        buildings = [None] * len(self.metergroups)
        
        if len(buildings) > 1:
            
            env = self.generate_multiple_building()
        
        else: 
            
            env = self.generate_single_building(self.metergroups[0],0)
        
        return env
                                
                                
    def save_environment(self, data, filename):
                                
        data.reset_index().to_feather(filename)
                                


def get_status(app, threshold, min_off, min_on):
    
    #creates a Boolean series to check the values over the threshold
    condition = app > threshold
    
    # Finds the indicies of changes in "condition" - those are True
    d = np.diff(condition)
    
    # Returns the indices of the elements that are non-zero. not false in our case
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns - this creates each consecutive index to be on the other side
    
    idx.shape = (-1,2)
    
    on_events = idx[:,0].copy()
    off_events = idx[:,1].copy()
    
    #checks if they are of even size 
    assert len(on_events) == len(off_events)

    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        
        #adds 1000 to the beginning
        off_duration = np.insert(off_duration, 0, 1000.)
        #keeps the indices where the off duration is less than min_off
        on_events = on_events[off_duration > min_off]
        
        #places the 1000 as the last element in off_duration
        off_events = off_events[np.roll(off_duration, -1) > min_off]
        assert len(on_events) == len(off_events)

        on_duration = off_events - on_events
        on_events = on_events[on_duration > min_on]
        off_events = off_events[on_duration > min_on]

    s = app.copy()
    #s.iloc[:] = 0.
    s[:] = 0.

    for on, off in zip(on_events, off_events):
        #s.iloc[on:off] = 1.
        s[on:off] = 1.
    
    return s                                
        