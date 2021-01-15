#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:32:16 2021

@author: matthewyeung
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#without using pytorch Dataloader module, use the dataset itself
def x_loader(dataset, timestep, start_from, batch_size, random_list, Shuffle = False):
    if Shuffle == True:
        dataset_ = dataset[random_list[0]:random_list[0] + timestep, :]
        for i in range(1,len(random_list)):
            dataset_ = torch.cat((dataset_, dataset[random_list[i]: random_list[i] + timestep, :]))
    else:        
        dataset_ = dataset[start_from:start_from + timestep, :]
        if batch_size > 1:
            for batch in range(1,batch_size):
                dataset_ = torch.cat((dataset_, dataset[start_from + batch:
                                                        start_from + batch + timestep, :]))
    
    return dataset_.view(batch_size,timestep,-1)

def y_loader(dataset_price, timestep, start_from, batch_size, random_list, Shuffle = False):
    if Shuffle == True:
        dataset_ = dataset_price[random_list[0] + timestep].view(1)
        for i in range(1,len(random_list)):
            dataset_ = torch.cat((dataset_, dataset_price[random_list[i] + timestep].view(1)),0)
    
    else:
        dataset_ = dataset_price[start_from + timestep].view(1)
        if batch_size > 1:
            for batch in range(1,batch_size):
                dataset_ = torch.cat((dataset_, dataset_price[start_from + timestep + batch].view(1)),0)
    return dataset_.view(batch_size)