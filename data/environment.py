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

from utils.path_finder import NILMTK_SOURCE, SCENARIOS
 

class Environment(object): 
    
    """
    Represents an environment to which an experiment is conducted.

    ...

    Attributes
    ----------
    scenario : int
        the scenario that defines the environment params
    mode : str
        the mode of the environment i.e train or test
    source : str
        the source filename of the environment's dataset  
    dataset : str
        the name of the environment's dataset i.e UK-DALE, REDD etc.
    buildings : list of ints
        the list of buildings used to create the environment
    sample_period : int
        the sample period of the data
        
    meter_groups : list of nilmtk.MeterGroups
    
    appliances : dict 
        the target appliances with their corresponding min-off duration, min-on duration, power threshold
    """
    
    def __init__(self, scenario=1, mode='train'):
        
        super().__init__()
        
        self.scenario = scenario
        self.mode = mode
        self.source = None
        self.dataset = None
        self.buildings = []
        self.sample_period = None  
        self.metergroups = []
        self.appliances = {}
        
    def configure_environment(self): 
        
        """Configures the environment as defined in the scenario configuration file."""
        
        filename = os.path.join(SCENARIOS,"scenario_{}.json".format(self.scenario))
        
        with open(filename) as file:
            conf = json.load(file)
        
        self.source = os.path.join(NILMTK_SOURCE,conf['{}_source'.format(self.mode)])
        self.dataset = DataSet(self.source).metadata['name']
        self.buildings = conf['{}_buildings'.format(self.mode)] 
        self.sample_period = conf['sample_period']
        
        self.metergroups = self.configure_metergroups(conf)
        self.appliances = self.configure_appliances(conf)
            
        
    def configure_metergroups(self, conf):
        
        """Configures the environment as defined in the scenario configuration file.
        
        Parameters
        ----------
        conf : dict, scenario configuration params

        Returns
        -------
        List of nilmtk.MeterGroups, each MeterGroup corresponds to a building.
        
        """
        
        windows = conf['{}_windows'.format(self.mode)]   
        
        metergroups = [None]*len(self.buildings)
        
        for i, building in enumerate(self.buildings) :
            
            dataset = DataSet(self.source)
            dataset.set_window(start= windows[i][0], end=windows[i][1])
            metergroups[i]= dataset.buildings[building].elec
            
        return metergroups
    

    def configure_appliances(self, conf):
        
        """Sets the min-off duration, min-on duration, power threshold for each target appliance nilmtk.ElecMeter.
        
        Parameters
        ----------
        conf : dict, scenario configuration params

        Returns
        -------
        dict,  the target appliances with their corresponding min-off duration, min-on duration, power threshold.
        
        """
        
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
            
        
    def get_chunk_activations(self, metergroup, appliance, timeframe):
        
        """Returns the activations of an appliance, i.e periods when the appliance is on.
        
        Parameters
        ----------
        metergroup : nilmtk.MeterGroup, contains the building meters
        appliance : str, appliance name
        timeframe : nilmtk.TimeFrame, the timeframe in which the function searches for the activations

        Returns
        -------
        list of pandas.Series,  each series contains one activation.
        
        """
        
        meter = metergroup[appliance]
        chunk = next(meter.power_series(sections=[timeframe]))
        
        activations = get_activations(chunk = chunk, 
                                      min_off_duration= meter.min_off_duration,     
                                      min_on_duration=meter.min_on_duration,
                                      on_power_threshold=meter.on_power_threshold,
                                      border=0)  
        
        return activations
        
    
    def align_mains_with_meter(self, metergroup, appliance, chunk, timeframe): 
        
        """Aligns the mains chunk with the activations of an appliance, for a specific timeframe. 
        
        This function is used to align the mains and appliance meter chunks even in-case there is 
        a misalignment in the datetime index.
        
        Parameters
        ----------
        metergroup : nilmtk.MeterGroup, contains the building meters
        appliance : str, appliance name
        chunk : pandas.DataFrame
        timeframe : nilmtk.TimeFrame, the timeframe that corresponds to the mains chunk indices

        Returns
        -------
        pandas.Series,  a series containing the appliance meter chunk.
        
        """
        
        chunk_activations = self.get_chunk_activations(metergroup, appliance, timeframe)

        # initialise the appliance as OFF 
        chunk[appliance] = 0
        
        if chunk_activations:  
            
            for i in range(len(chunk_activations)):
             
                # the appliance is ON for the mains indices that are inside the chunk activation
                mask = (chunk.index >= chunk_activations[i].index[0]) & (chunk.index <= chunk_activations[i].index[-1])
                chunk.loc[mask, appliance] = 1

        return chunk[appliance] 
    
    
    def generate_single_building(self, building, i):
        
        """Generates the dataset for a single building. 
         
        Parameters
        ----------
        building : nilmtk.MeterGroup, contains the building meters
        i : int, building index

        Returns
        -------
        pandas.DataFrame,  a dataframe containing the aligned mains and appliance meters data.
        
        """
        
        print("Preparing {} building {} {}ing set ...".format(self.dataset, self.buildings[i], self.mode)) 
        

        mains = building.mains()
        
        # Locate the good sections (where the sample period is <= max_sample_period) of the aggregate (mains) meter 
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
                
                chunk[appliance] = self.align_mains_with_meter(building, appliance, chunk, chunk_timeframe)
            
            sections.append(chunk)

        if sections:    
            return pd.concat(sections)
      
        else:
            return pd.DataFrame(columns=["site meter"].append(self.appliances))
        
    
    def generate_multiple_building(self):
        
        """Generates the dataset for multiple building. 


        Returns
        -------
        pandas.DataFrame,  a dataframe containing the aligned mains and appliance meters data.
        
        """

        
        buildings = [None] * len(self.metergroups)
        
        for i, building in enumerate(self.metergroups):
            
            buildings[i] = self.generate_single_building(building, i)
            
        
        if any(k is not None for k in buildings): 
            
            # buildings are concatenated serially (no shuffling). 
            
            data = pd.concat(buildings)
            return data
        
        else:
            return pd.DataFrame(columns=["site meter"].append(self.appliances))
            

    
    def generate_environment(self):
        
        """Generates the environment's dataset. 


        Returns
        -------
        pandas.DataFrame,  a dataframe containing the aligned mains and appliance meters data.
        
        """
        
        buildings = [None] * len(self.metergroups)
        
        if len(buildings) > 1:
            
            env = self.generate_multiple_building()
        
        else: 
            
            env = self.generate_single_building(self.metergroups[0],0)
        
        return env
                                
                                
    def save_environment(self, data, filename):
                                
        data.reset_index().to_feather(filename)
                                


def get_on_off(chunk, on_power_threshold, min_off_duration=0,  min_on_duration=0):
    
    """Defines the chunk on, off states based on the on_power_threshold, min_off_duration,  min_on_duration. 

    Parameters
    ----------
    
    chunk : numpy.array
    on_power_threshold : int or float
        Watts
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.

    Returns
    -------
    numpy.array,  the chunk's on, off states

    """
    
    when_on = chunk > on_power_threshold
    
    state_changes = np.diff(when_on)
    idx, = state_changes.nonzero() 

    # We need to start after the change in "when_on". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if when_on[0]:
        idx = np.r_[0, idx]

    if when_on[-1]:
        idx = np.r_[idx, condition.size]

    
    idx.shape = (-1,2)
    
    switch_on_events = idx[:,0].copy()
    switch_off_events = idx[:,1].copy()
    
    #checks if they are of even size 
    assert len(switch_on_events) == len(switch_off_events)

    if len(switch_on_events) > 0:
        
        off_duration = switch_on_events[1:] - switch_off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000.)

        switch_on_events = switch_on_events[off_duration > min_off_duration]
        switch_off_events = switch_off_events[np.roll(off_duration, -1) > min_off_duration]
        
        assert len(switch_on_events) == len(switch_off_events)

        on_duration = switch_off_events - switch_on_events
        switch_on_events = switch_on_events[on_duration > min_on_duration]
        switch_off_events = switch_off_events[on_duration > min_on_duration]

    s = chunk.copy()
    s[:] = 0.

    for on, off in zip(switch_on_events, switch_off_events):
        s[on:off] = 1.
    
    return s                                
        