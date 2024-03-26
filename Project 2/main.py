## Inlcudes
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timezone
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle


## Features dependign on areas
area_1 = ['timestamp', 'NO1_consumption', 'NO1_temperature']
area_2 = ['timestamp', 'NO2_consumption', 'NO2_temperature']

fileName = 'consumption_and_temperatures.csv'


#######################
##   Preparing data  ##
#######################
def read_data(filepath, features):
    # This is the function we discussed earlier
    data = pd.read_csv(filepath)
    if all(column in data.columns for column in features):
        selected_df = data[features]
    else:
        raise ValueError("One or more selected columns are not in the file")
    return selected_df

def split_timestamp(data, timestamp_column = 'timestamp'):
    # Ensure the timestamp column is in datetime format
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], utc=True)
    
    # Calculate new features
    data['time_of_day'] = data[timestamp_column].dt.hour 
    data['day_of_week'] = data[timestamp_column].dt.weekday
    data['day_of_year'] = data[timestamp_column].dt.dayofyear
    
    # Drop the original timestamp column
    data.drop(columns=[timestamp_column], inplace=True)

    return data

def normalize(data, save_path='normalization_LSTM_1.pkl'):
    mean = data.mean()
    std = data.std()
    
    normalized_data = (data - mean) / std

    # Saving mean and std to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    return normalized_data



def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length), 1:]
        y = data.iloc[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
    
def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the split ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1
    
    # Calculate the indices for splitting
    total_samples = X.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split the data
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test




def preprocessing(fileName, area, seq_length= 15, model_type='LSTM'):
    # Reading from file
    data = read_data(fileName, area)

    # Getting time on right format
    data = split_timestamp(data)

    #Normalizing the data
    data = normalize(data)

    data = data.rename(columns={area[1]: 'target'})

    if model_type == 'LSTM':
        X, y = create_sequences(data, seq_length)
        
        # Split the data (adjust the split_data function or manually split here)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
        
        # Convert the datasets into PyTorch tensors
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)
        X_val = torch.Tensor(X_val)
        y_val = torch.Tensor(y_val)
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test)

        # Directly return the tensors for training, validation, and testing
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        # TODO: ADD CNN and (Transformer or feed-forward)
        pass



####################
##    Training    ##
####################
def train(model, X_train, y_train, X_val, y_val, epochs = 10, save=False, save_model_path=None, save_history_path=None):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=24)
    history = {'train': {'mse': [], 'mae': []}, 'val': {'mse': [], 'mae': []}}

    for epoch in range(epochs):
        model.train()
        train_mse_accum = 0.0
        train_mae_accum = 0.0
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse_accum += loss_fn(y_pred, y_batch).item() * X_batch.size(0)
            train_mae_accum += torch.abs(y_pred - y_batch).sum().item()

        train_mse = train_mse_accum / len(X_train)
        train_mae = train_mae_accum / len(X_train)

        history['train']['mse'].append(train_mse)
        history['train']['mae'].append(train_mae)

        # Validation
        model.eval()
        val_mse_accum = 0.0
        val_mae_accum = 0.0
        with torch.no_grad():
            y_pred = model(X_val)
            y_pred = y_pred.squeeze()
            val_mse_accum += loss_fn(y_pred, y_val).item() * X_val.size(0)
            val_mae_accum += torch.abs(y_pred - y_val).sum().item()

        val_mse = val_mse_accum / len(X_val)
        val_mae = val_mae_accum / len(X_val)

        history['val']['mse'].append(val_mse)
        history['val']['mae'].append(val_mae)

        print(f"Epoch {epoch} | train MSE {train_mse:.4f} | val MSE {val_mse:.4f} | train MAE: {train_mae:.4f} | val MAE {val_mae:.4f}")

    if save:
        torch.save(model.state_dict(), save_model_path)
        pickle.dump(history, open(save_history_path, 'wb'))



    return model, history

####################
##     Plots      ##
####################
def plot_train_losss(history):
    epochs = range(1, len(history['train']['mse']) + 1)

    # Plot training and validation MSE
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train']['mse'], label='Training MSE')
    plt.plot(epochs, history['val']['mse'], label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Plot training and validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train']['mae'], label='Training MAE')
    plt.plot(epochs, history['val']['mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()




####################
##     MODELS     ##
####################
class LstmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        self.batch_norm = torch.nn.BatchNorm1d(num_features=100)
        x = self.linear(x)
        return x
    

X_train, y_train, X_val, y_val, X_test, y_test = preprocessing(fileName, area_1, 15)

LSTM = LstmModel()
model, history = train(LSTM, X_train, y_train, X_val, y_val, save=False, save_model_path='./models/LSTM_n01.pth', save_history_path='./models/LSTM_n01_history.pkl')





