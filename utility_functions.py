#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:25:33 2021

@author: matthewyeung
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score 

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#learning curve
def learning_curve(avg_losses, name):
    plt.figure(figsize = (15,8))
    plt.plot(avg_losses, color='g' ,label='Loss')
    plt.grid()
    plt.title(name + ' - Training Loss')
    plt.legend()
    plt.savefig(name + '_Training_Loss.png', dpi = 100)
    plt.show()

#learning curve
def Test_loss(test_loss, name):
    plt.figure(figsize = (15,8))
    plt.plot(test_loss, color='g' ,label='Loss')
    plt.grid()
    plt.title(name + ' - Testing Loss')
    plt.legend()
    plt.savefig(name + '_Testing_Loss.png', dpi = 100)
    plt.show()

# Training loss vs Validation loss
def train_val_loss(loss_epoch,mse, name):
    fig,ax = plt.subplots(figsize = (15,8))
    plt.grid()
    # make a plot
    ax.plot(loss_epoch, color="blue", label='Training')
    plt.legend()
    # set x-axis label
    ax.set_xlabel("Epoch",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Loss",color="blue",fontsize=14)
    # set title
    ax.set_title(name+ ' - Training Loss vs Validation Loss')
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(mse , color="orange", label='Validation' )
    plt.legend()
    ax2.set_ylabel("Loss",color="orange",fontsize=14)
    plt.savefig(name + '_train_val_loss.png', dpi = 100)
    plt.show()

#predict and target curve
def predict_target(predict_data, target_data, name):
    plt.figure(figsize = (15,8))
    plt.plot(predict_data, label='Predict')
    plt.plot(target_data, label='target')
    plt.grid()
    plt.title(name + ' - Distribution of Predict and Target Value')
    plt.legend()
    plt.savefig(name + '_predict_target_value.png', dpi = 100)
    plt.show()

#Buy and sell with price
def buy_price(buysell_val, name):
    plt.figure(figsize = (15,8))
    plt.grid()
    plt.plot(buysell_val['price'], color='b' ,label='Price')
    plt.plot(buysell_val['new_buy'], color='g' ,label='Buy', marker = 'x')
    plt.title(name + ' - Buy and Sell action with Price',fontsize = 14)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(name + '_testing_action_price.png', dpi = 100)
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clone_detach(cum_list):
    cum_list_1 = []
    for i in cum_list:
        cum_list_1.append(i.cpu().clone().detach().numpy())
    return cum_list_1

def to_df(predict_list, target_list, thresold):
    df1 = pd.DataFrame(predict_list, columns = ['predict'])
    df1['y_predict'] = 0
    for i in range(len(df1)):
        if df1['predict'].iloc[i] > thresold:
            df1['y_predict'].iloc[i] = 1
    df2 = pd.DataFrame(target_list, columns = ['y_target'])
    df = pd.concat([df1,df2], axis = 1)
    y_predict = df1['y_predict'].values
    y_target = df2['y_target'].values
    return df, y_predict, y_target

def buysell_price(price, buysell):
    buysell = pd.DataFrame(buysell, columns = ['Buy_sell'])
    price = pd.DataFrame(price, columns = ['price'])
    df = pd.concat([price , buysell], axis =1)
    df['new_buy'] = df['Buy_sell']*df['price']
    for i in range(len(df)):
        if df['Buy_sell'].iloc[i] == 0:
            df['new_buy'].iloc[i] = None
    df = df[['price','new_buy']]
    return df

def scores(y_true, y_predict):
    recall = recall_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    accuracy = accuracy_score(y_true,y_predict)
    confusion = confusion_matrix(y_true, y_predict)