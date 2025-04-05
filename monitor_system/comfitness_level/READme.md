# comfitness_level
## Introduction 
This file is for comfitness advice. The key concept is using [Heat Index suggested by NOAA](https://www.wpc.ncep.noaa.gov/html/heatindex.shtml). 
It takes **temperature in Celsius °C** and **relative humidity level in %**, and gives warning text message for users under different criteria.  

Key file are comfitness_level_NN.py and model file. 

test_comfitness.py indicates how you can implement this neuron network. 

## How this was achieved. 
In data file, using comfort_level function in comfitness_level.py to generate range of random data sets with temperature and humidity. 
This will give us training sets in a csv file. comfitness_level_NN.py will load the data set and used it for training the neuron network with 2 linear layers. 
Each layer will be using ReLu activation function and this is known as feedforward netowrk. 
During the training process, it will use the cross entropy to calculate the losses and the learning rate is adjust by Adam optimizer, one of the most commonly used optimiser for LR.
It will print the losses for every 10 epoches and we sugeest at least 500 epoches or you can run the file multiple times before a satisified losses level is given. 

All the data and neuron network itself will be saved in model file. 
