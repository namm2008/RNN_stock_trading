# RNN_stock_trading
Using Recurrent Neural Network (GRU and LSTM) to predict the stock price movement with Pytorch 

## Instruction
This repository aimed at applying a stacked GRU / LSTM to generate stock trading signals in order to make profit from the stock markets. The model was written in Python and the neural network was coded with Pytorch. You can clone all the files and run the 'mainloop.py' for result generation and testing. The hyperparameters can be adjusted in this file as well. All the result will be saved for further analysis. The stock data was downloaded by the 'Yahoo Finance' API in the codes. You need to choose a stock by there 'ticker' and the period that you want to investigate. The train test split can be adjusted in the mainloop.py file. 

## Model Training
### Data Loader:
The downloaded stcok price data is then pass to generate the technical anaylsis tools including EMA short lines, EMA long lines, RSI, and MACD for features input. In the code, as RNN model is a supervised learning model, a label column has to be added as the target of the output. In order to signify the trading signals, a binary label with 0 and 1 was used. 0 represented holding cash or selling a stock; where 1 represented buying a stock or holding a stock. The labels depended on the percentage change of the current price to previous price (‘pct_change’). Also, a 5-day average percentage change of price was used instead of one day change. Given the fact that a shorter window are prone to short term noise, 5-day average price were introduced to calculate the signal label. The data loader allowed the model to retrieve a series of training data as input and the label as the output.

### Stacked GRU/LSTM
As both networks work and be coded similarly, here GRU was used as example to explain the mechanism. A 3-layer stacked GRU was used. The input size of GRU aligned with the original shape so that no reshaping was needed. The hidden dimensions in all layers was set to 25. To begin with, a hidden state had to be initialized as another input to the network. This hidden state was initialized with all zeros and shape equal to (layer number, Batch size, Hidden dimension). The output from the GRU layers was of shape equal to (Batch size, hidden dimension x timestep). Firstly, Dropout with ratio 0.5 was carried out to avoid overfitting problem. In order to further extract the information learned from the GRU, a hidden linear layer with 8 neurons and ReLu activation was used. The shape needed to be converted to 1 output units and sigmoid function was applied.

### Loss Function and Optimization
The loss function implemented was the Mean Squared Errors (MSE) between the predicted values and the target values.\
**Loss=∑(Y_i- Y'_i)^2 / n**\
where Y_i was the predicted values, Y'_i was the target value and n is the batch size. ADAM optimization algorithm was used. 

### Validation
The training was carried out with validation applied. To avoid overfitting problem, the training was kept on until the stop criteria met. There were training loss tolerance and validation loss tolerance. When the number of epoches for the training loss stop reducing which exceed the tolerance number, the stop criteria were triggered. 

## Model Testing
In the testing phase, the models were kept training as the same practice in the RL models. Most of the settings were kept the same as those in the training phase. For the output, as sigmoid activation function was applied to it, the output was in the range of [0,1]. In order to perform the action, a threshold number was implemented. If the output was greater than the threshold, the output was treated as 1, vice versa. \
**Thresold=(maxY_i  + min⁡〖Y_i 〗)/2** \
where Y_i was the predicted values. Finally, the batch size was reduced to 15 in order to keep the network up to date. 



