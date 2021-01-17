#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: matthewyeung
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Recurrent network (GRU)
class GRU(nn.Module):
    def __init__(self, output_size, feature_number, hidden_dim, n_layers, seq_length, drop_prob=0.5):
        super(GRU, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        self.gru = nn.GRU(feature_number, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(hidden_dim*seq_length, 8)
        
        self.fc2 = nn.Linear(8, output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)

        gru_out, hidden = self.gru(x, hidden)
        gru_out = gru_out.contiguous().view(batch_size, -1)
        
        out = self.dropout(gru_out)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.sigmoid(self.fc2(out))

        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

#Recurrent network (LSTM)
class LSTM(nn.Module):
    def __init__(self, output_size, feature_number, hidden_dim, n_layers, seq_length, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(feature_number, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(hidden_dim*seq_length, 8)
        
        self.fc2 = nn.Linear(8, output_size)

        
    def forward(self, x, hidden):
        batch_size = x.size(0)

        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.sigmoid(self.fc2(out))

        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
