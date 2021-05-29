# An Experiment Framework for Multi-label NILM with Neural Networks

This repository defines an experiment framework for multi-label classification on NILM and is used to create and evaluate a baseline neural network architecture set. 


## Data 

This project supports the following datasets, using the [NILMTK](https://github.com/NILMTK/NILMTK) toolkit to read and preprocess the power recordings:

- [UKDALE](https://jack-kelly.com/data/), which records the power demand from five houses in the UK. 
- [REDD](http://redd.csail.mit.edu/), which contains several weeks of power data from 6 houses in the USA. 
- [iAWE](https://iawe.github.io/), which contains power data from 1 house in India. 


## Neural Network Architectures

To be analysed. 

- Fully Convolutional Network (**FCN**), from [Deep learning for time series classification: a review](https://link.springer.com/article/10.1007/s10618-019-00619-1), Fawaz et al. (2019).
- Residual Network (**ResNet**), from [Deep learning for time series classification: a review](https://link.springer.com/article/10.1007/s10618-019-00619-1), Fawaz et al. (2019).
- Convolutional GRU (**ConvGRU**), from [A Dual-input Multi-label Classification Approach for NILM via Deep Learning](https://ieeexplore.ieee.org/document/9161776/), Çimen et al. (2020). 
- Convolutional LSTM (**ConvLSTM**), from [A Dual-input Multi-label Classification Approach for NILM via Deep Learning](https://ieeexplore.ieee.org/document/9161776/), Çimen et al. (2020). 
- Fully Convolutional Autoencoder (**FCN-AE**), a simplification, adaptation from [Non Intrusive Load Disaggregation by CNN and Multilabel Classification](https://www.mdpi.com/2076-3417/10/4/1454), Massidda et al. (2020). 


## Experiment Framework 

The experiment framework scenarios used in this project are an adaptation of the following publication: 

> Symeonidis, Nikolaos & Nalmpantis, Christoforos & Vrakas, Dimitris. (2019). A Benchmark Framework to Evaluate Energy Disaggregation Solutions. 10.1007/978-3-030-20257-6_2. 


## Project Structure 

The project structure is defined as follows: 

- 📂 **data:** Includes modules related to data e.g. loading data using **NILMTK**, detecting appliance activations and aligning meters.
    - **environment:** defines the environment params for the training, validation and test sets.
    - **generator:** defines 2 wrapper functions used to load the data for the experiments, **Seq2Seq** and **Seq2Point**. 
- 📂 **models:** Contains 5 neural network architectures used in the experiments
- 📂 **experiments:** Includes modules to configure and run the 4 experiment scenarios
    - **experiments:** run experiments, save detailed results
    - **metrics:** defines the multi-label metrics and reports
- 📂 **results:** Results of the experiments are saved in this directory.
- 📂 **utils:** Includes the utility functions
    - **path_finder:** defines a simple path manager 


## Requirements 

The dependecies of this code are listed as follows: 

- python>=3.6
- Cython>=0.27.3
- bottleneck>=1.2.1
- numpy>=1.13.3
- numexpr>=2.6.4
- pytables
- pandas>=0.25.3,<1.0
- matplotlib>=3.1.0,<3.2.0
- networkx==2.1
- scipy>=1.0.0
- scikit-learn>=0.21.2
- jupyter
- ipython
- ipykernel
- nose
- coverage
- pip
- psycopg2
- coveralls
- nilm_metadata
- nilmtk
- hmmlearn
- pytorch
- pyarrow
- seaborn
