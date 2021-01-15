#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:34:49 2021

@author: matthewyeung
"""

import math
import random
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
from collections import namedtuple
from itertools import count
from PIL import Image
from IPython.display import clear_output
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score 

from GRU_LSTM_network import GRU, LSTM
import backtest
import dataloader
import trading_environment
import transformation
import utility_functions

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize all the hyperparameters
BATCH_SIZE = 64
learning_rate = 0.0001
cost_percent = 0.25
commission = 0.25

output_size = 1
timestep = 20
n_layers = 3
hidden_dim = 32

num_epoch = 200
val_batch = 15
tolerance = 30
loss_tolerance = 30
stepfinished = 0
Shuffle_dataset = False

'''
model_list = ['RNN_GRU', 'RNN_LSTM']
data_feature_list = ['increasing', 'decreasing', 'normal']
data_time_data_list = ['noDT', 'DT']
'''

model_list = ['RNN_GRU']
data_feature_list = ['increasing']
data_time_data_list = ['noDT']

#For loop to loop all the models for GRU and LSTM models
for model in model_list:
    for data_feature in data_feature_list:
        for data_time_data in data_time_data_list:
            name = model + '_' + data_feature + '_' + data_time_data
            print(name)
            
            if data_feature == 'increasing':
                start = '2006-01-01'
                end = '2018-08-30'
                ticker = 'AMZN'
            elif data_feature == 'decreasing':
                start = '1998-01-01'
                end = '2019-12-31'
                ticker = 'F'
            elif data_feature == 'normal':
                start = '2003-01-01'
                end = '2018-12-31'
                ticker = 'T'                    
            ema_short = 9
            ema_long = 40
                
            if data_time_data == 'DT':
                data_time = True
                col_list_minmax = ['Close','Volume','high_low', 'ema_st','ema_lg', 'rsi','day', 'daytime']
            elif data_time_data == 'noDT':
                data_time = False
                col_list_minmax = ['Close','Volume','high_low', 'ema_st','ema_lg', 'rsi']
            df = stock_dataset_dl(ticker, start, end, ema_short, ema_long, data_time)
            
            #Train-Test Split
            train_test_ratio = 0.10
            dataset_len = len(df)
            test_length = round(train_test_ratio*dataset_len)
            df_test = df.iloc[-test_length:,:].reset_index(drop=True)
            df_train = df.iloc[:(-test_length),:]
            
            # minmaxstandardize
            df_test = interpolation(df_test, df_train, col_list_minmax)
            df_train = minmaxstandardized(df_train, col_list_minmax)            
            
            # delete row 0 to 4 for rsi=0
            df_train.drop([0,1,2,3],inplace = True)    
            df_test.drop([len(df_test)-1],inplace = True)
            
            #reset index
            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)
            
            #Define dataset and testset
            X_train = torch.tensor(df_train.iloc[:,1:-1].values, dtype=torch.float32, device=device)
            X_test = torch.tensor(df_test.iloc[:,1:-1].values, dtype=torch.float32, device=device)
            y_train = torch.tensor(df_train.iloc[:,-1].values, dtype=torch.float32, device=device)
            y_test = torch.tensor(df_test.iloc[:,-1].values, dtype=torch.float32, device=device)
            
            #Initialize the initial features
            feature_number = torch.tensor((X_train.shape[1]), device = device, dtype=torch.int32)
            dataset_len = len(X_train) - timestep - BATCH_SIZE - 5
            
            if model == 'RNN_GRU':
                #Inilialize the policy Net and Target Net
                rnnmodel = GRU(output_size, feature_number, hidden_dim, 
                                      n_layers, timestep, drop_prob=0.5).to(device)
            elif model == 'RNN_LSTM':
                rnnmodel = LSTM(output_size, feature_number, hidden_dim, 
                                      n_layers, timestep, drop_prob=0.5).to(device)
            #Loss Function
            loss_function = nn.MSELoss()
            
            #Define the Optimizer
            optimizer = optim.Adam(rnnmodel.parameters(),lr=learning_rate)
            
            # Count parameters:
            num_parameter = count_parameters(rnnmodel)
            
            #Training Loop:
            mse = []
            average_loss = []
            ave_loss = 0
            loss_epoch = []
            for epoch in range(num_epoch):
                epoch_losss = 0
                for i in range(0,dataset_len,BATCH_SIZE):
        
                    hidden = rnnmodel.init_hidden(BATCH_SIZE)


                    if Shuffle_dataset == True:
                        random_list = np.random.choice(dataset_len, BATCH_SIZE)
                        X_data = x_loader(X_train, timestep, rand, BATCH_SIZE, random_list, Shuffle_dataset)
                        targets = y_loader(y_train, timestep, rand, BATCH_SIZE, random_list, Shuffle_dataset)
                    else:
                        X_data = x_loader(X_train, timestep, i, BATCH_SIZE, None, Shuffle_dataset)
                        targets = y_loader(y_train, timestep, i, BATCH_SIZE, None, Shuffle_dataset)

                    # Step 3. Run our forward pass.
                    predict,_ = rnnmodel(X_data, hidden)

                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = loss_function(predict, targets)
                    loss.backward()
                    for param in rnnmodel.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()
                    losss = loss.item()
                    ave_loss += losss
                    epoch_losss += losss
        


        
                    if i % (BATCH_SIZE*3) == 0:
                        #Save Model
                        save_name = name
                        torch.save({'rnnmodel': rnnmodel.state_dict(),
                                    'optimizer': optimizer.state_dict()}, save_name + '.pt')
                        average_loss.append(ave_loss/3)
                        ave_loss = 0
        
                loss_epoch.append(epoch_losss)
    
                #Validation:
                with torch.no_grad():
                    rand = np.random.choice(len(X_test) - timestep - val_batch - 1, 1)[0]
                    X_data_test = x_loader(X_test, timestep, rand, val_batch, None, False)
                    targets = y_loader(y_test, timestep, rand, val_batch, None, False)
                    hidden = rnnmodel.init_hidden(val_batch)
                    predicts,_ = rnnmodel(X_data_test,hidden)
                    MSE = (predicts - targets).mean()
                    mse.append(MSE)

    
                if MSE == np.min(mse) or epoch_losss == np.min(loss_epoch):
                    save_name = name + '_best'
                    torch.save({'rnnmodel': rnnmodel.state_dict(),
                                'optimizer': optimizer.state_dict()}, save_name + '.pt')
                
                training_episode = epoch
                #Break if loss attemp min for tolerance = loss_tolerance
                if epoch >= loss_tolerance:
                    if np.min(loss_epoch) == loss_epoch[-loss_tolerance]:
                        break
                #Break if Validation loss attemp min for tolerance = tolerance
                if epoch >= tolerance:
                    if np.min(mse) == mse[-tolerance]:
                        break
                
            stepfinished += 1
            print('Complete Training ' + name + ' - finished : ', stepfinished)
            print('training epoch = ', training_episode)
            
            #print learning curve
            learning_curve(average_loss, name)
            
            #print train_val curve
            train_val_loss(loss_epoch, mse, name)
            
            #save the training data
            np.savetxt(name + "_train_val_loss.csv", np.column_stack((loss_epoch, mse)), delimiter=",", fmt='%s')

            #Initialize the initial features
            feature_number = torch.tensor((X_test.shape[1]), device = device, dtype=torch.int32)
            testset_len = len(X_test) - timestep - 1 - 5

            #Load Model
            save_name = name + '_best'
            checkpoint = torch.load(save_name + '.pt')
            if model == 'RNN_GRU':
                #Inilialize the policy Net and Target Net
                rnnmodel = GRU(output_size, feature_number, hidden_dim, 
                                      n_layers, timestep, drop_prob=0.5).to(device)
            elif model == 'RNN_LSTM':
                rnnmodel = LSTM(output_size, feature_number, hidden_dim, 
                                      n_layers, timestep, drop_prob=0.5).to(device)
            rnnmodel.load_state_dict(checkpoint['rnnmodel'])
            
            #testing
            predict_data = []
            target_data = []
            test_loss = []
            for i in range(0,testset_len,1):

                hidden = rnnmodel.init_hidden(1)

                X_data = x_loader(X_test, timestep, i, 1, None, False)
                targets = y_loader(y_test, timestep, i, 1, None, False)

                # Step 3. Run our forward pass.
                predict,_ = rnnmodel(X_data, hidden)
                predict_data.append(predict[0])
                target_data.append(targets)

                #  calling optimizer.step()
                loss = loss_function(predict, targets)
                loss.backward()
                for param in rnnmodel.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                losss = loss.item()
                test_loss.append(losss)
            
            predict_data = clone_detach(predict_data)
            target_data = clone_detach(target_data)
            price_data = df_test['price'][timestep:-(6)].values
            
            thresold = (np.max(predict_data) + np.min(predict_data))/2
            df_describe, y_predict, y_target = to_df(predict_data, target_data, thresold)
            
            df_describe.to_csv(name + 'predict_target.csv')
            
            Test_loss(test_loss, name)
            
            predict_target(predict_data, target_data, name)
            
            confuse = confusion_matrix(y_target, y_predict)
            np.savetxt(name + '_confustion.csv', confuse, delimiter=",", fmt='%s')
            
            buysell_val = buysell_price(price_data, y_predict)
            
            buy_price(buysell_val, name)
            
            df_each_trade_pnl = model_strategy(buysell_val, commission, name, training_episode, num_parameter)
            
            df_cum_price_val = cum_perform(price_data)
            
            return_underlying(df_each_trade_pnl, df_cum_price_val, name)
            
            print('Finished ' + name)