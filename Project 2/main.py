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

#######################
##   Preparing data  ##
#######################
def prepare_data(X, Y, mode='LSTM'):
    
    if mode == 'LSTM':
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float)
        Y_tensor = torch.tensor(Y, dtype=torch.float)

    elif mode == 'OtherModel':
        # Apply transformations specific to OtherModel
        pass
    
    return X_tensor, Y_tensor

def denormalize(predictions, load_path='mean_values/normalization_LSTM_1.pkl'):
    # Load the mean and std values
    with open(load_path, 'rb') as f:
        normalization_params = pickle.load(f)
    mean = normalization_params['mean'][0]
    std = normalization_params['std'][0]
    
    # Reverse the normalization process
    denormalized_data = [p * std + mean for p in predictions]

    return denormalized_data

####################
##    Training    ##
####################
import torch
import torch.optim as optim
import torch.utils.data as data
import pickle

def train(model, X_train, y_train, X_val, y_val, epochs=10, save=False, save_model_path=None, save_history_path=None):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=24)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), batch_size=24)
    history = {'train': {'mse': [], 'mae': []}, 'val': {'mse': [], 'mae': []}}

    for epoch in range(epochs):
        model.train()
        train_mse_accum = 0.0
        train_mae_accum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch)
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
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze()
                val_mse_accum += loss_fn(y_pred, y_batch).item() * X_batch.size(0)
                val_mae_accum += torch.abs(y_pred - y_batch).sum().item()

        val_mse = val_mse_accum / len(X_val)
        val_mae = val_mae_accum / len(X_val)
        
        history['val']['mse'].append(val_mse)
        history['val']['mae'].append(val_mae)

        print(f"Epoch {epoch} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

    if save and save_model_path and save_history_path:
        torch.save(model.state_dict(), save_model_path)
        with open(save_history_path, 'wb') as f:
            pickle.dump(history, f)

    return model, history



####################
##  Predictions   ##
####################
def predict_next_24_hours(model, day_1, day_2_features):
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []

    # Convert day_1 to a 3D tensor matching the model's expected input shape [1, seq_length, features]
    current_sequence = day_1.unsqueeze(0)

    # Ensure day_2_features is also a 3D tensor [1, seq_length, features-1]
    day_2_features = day_2_features.unsqueeze(0)

    for i in range(day_2_features.shape[1]):  # Iterate through each step in day_2
        with torch.no_grad():
            # Make prediction based on the current sequence
            predicted_y = model(current_sequence).squeeze(-1)  # Model output assumed to be [1, 1]
            predictions.append(predicted_y.item())

            # Shift the sequence to the left to make room for the next features from day_2
            current_sequence = torch.roll(current_sequence, -1, dims=1)
            # Update the first feature (previous y) 
            current_sequence[:, -1, 0] = predicted_y

            # Update the rest of the features 
            if i < day_2_features.shape[1] - 1:  
                current_sequence[:, -1, 1:] = day_2_features[:, i, :]

    return predictions

def predict_next_24_CNN(model, day_1, day_2):
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []

    # Add a batch dimension and transpose to get [1, features, seq_length]
    current_sequence = day_1.unsqueeze(0).transpose(1, 2)

    # day_2_features is 2D: [seq_length, features-1], lacking the target feature
    # Add a batch dimension and transpose to get [1, features-1, seq_length]
    day_2_features = day_2.unsqueeze(0).transpose(1, 2)

    for i in range(23):  # Predict the next 24 hours
        with torch.no_grad():
            # Make prediction based on the current sequence
            predicted_y = model(current_sequence).squeeze()  # Adjust depending on model output shape
            predictions.append(predicted_y.item())

            # Roll the sequence to shift time steps
            current_sequence = torch.roll(current_sequence, -1, dims=2)
            
            
            # Insert the predicted value as the first feature in the last time step
            current_sequence[:, 0, -1] = predicted_y  # Assuming the target is the first feature
            
            # If there are additional features from day_2 to add, do so here
            if i < day_2_features.shape[2] - 1:  # Ensure we don't go out of bounds
                # Update the features for the next time step, excluding the first feature (target)
                current_sequence[:, 1:, -1] = day_2_features[:, :, i+1]

    return predictions

def predict_next_24_feed(model, day_1, day_2_features):
    model.eval()  # Set model to evaluation mode
    predictions = []

    # Flatten day_1 data for the initial input
    current_input = day_1.view(-1).unsqueeze(0)  # Flattening and adding batch dimension

    for i in range(23):  # Predict the next 24 hours
        with torch.no_grad():
            predicted_y = model(current_input).squeeze(-1)
            predictions.append(predicted_y.item())

            # Update current_input for the next prediction:
            if i < 23:  # For all but the last iteration
                # Slide current_input to discard the earliest time step
                current_input = current_input[:, 5:]  # Assuming 5 features per time step

                # Add the new prediction and the next available features from day_2
                # Note: We need to handle the missing target value in day_2_features
                next_features = torch.cat((predicted_y.view(1, 1), day_2_features[i].view(1, -1)), dim=1)
                current_input = torch.cat((current_input, next_features), dim=1)

    return predictions

def update_input_with_prediction(predicted_y, next_input_features):
    # This function should create the new input array for the next prediction
    # It needs to combine the predicted_y with the features from day_2
    # The specific implementation will depend on how your features are structured and used by the model
    # This is a placeholder for conceptual guidance
    new_input = torch.tensor([predicted_y] + next_input_features.tolist()).unsqueeze(0)
    return new_input






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

def plot_prediction(observed, real_values, predictions):
    plt.figure(figsize=(10, 6))

    # Plotting target values
    plt.plot(observed, 'g-', label='Target Values')

    # Plotting predicted values
    plt.plot(range(len(observed), len(observed) + len(predictions)), predictions, 'rx', label='Predicted Values')

    # Plotting real values with 'o' marker
    plt.plot(range(len(observed), len(observed) + len(real_values)), real_values, markersize=5, marker='o', linestyle='None', label='Real Values')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Target, Predicted, and Real Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_comparisons(observed, real_values, predictions, predictions_2):
    plt.figure(figsize=(10, 6))

    # Plotting target values
    plt.plot(observed, 'g-', label='Target Values')

    # Plotting predicted values
    plt.plot(range(len(observed), len(observed) + len(predictions)), predictions, 'rx', label='Predicted')
    plt.plot(range(len(observed), len(observed) + len(predictions_2)), predictions_2, 'yx', label='Predicted diffrent model')

    # Plotting real values with 'o' marker
    plt.plot(range(len(observed), len(observed) + len(real_values)), real_values, markersize=5, marker='o', linestyle='None', label='Real Values')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Target, Predicted, and Real Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error(array1, array2, array3):
    # Assuming array1, array2, and array3 are numpy arrays with shape (23,)
    
    # Generate a sequence of 23 time steps
    time_steps = np.arange(1, 24)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each array
    plt.plot(time_steps, array1, label='CNN', marker='o', linestyle='-', color='blue')
    plt.plot(time_steps, array2, label='Feed forward', marker='x', linestyle='--', color='red')
    plt.plot(time_steps, array3, label='LSTM', marker='s', linestyle='-.', color='green')
    
    # Adding some plot decorations
    plt.title('Comparison of Three Arrays')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_all_loss(loss1, loss2, loss3, labels=['Loss 1', 'Loss 2', 'Loss 3']):
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(loss1) + 1)

    plt.plot(epochs, loss1, label=labels[0], marker='o', linestyle='-', color='blue')
    plt.plot(epochs, loss2, label=labels[1], marker='x', linestyle='--', color='red')
    plt.plot(epochs, loss3, label=labels[2], marker='s', linestyle='-.', color='green')

    plt.title('Comparison of Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()




####################
##     MODELS     ##
####################
class LstmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(50, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        self.batch_norm = torch.nn.BatchNorm1d(num_features=100)
        x = self.linear(x)
        return x
    
class CnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1d parameters: in_channels, out_channels, kernel_size
        self.conv1 = torch.nn.Conv1d(in_channels=5, out_channels=64, kernel_size=23, padding='same')
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=24, padding='same')
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x):
        # PyTorch Conv1d expects input shape of (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(23*5, 128)  # Input layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 1)    # Output layer for regression

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, direct output for regression
        return x


if __name__ == '__main__':
    ##Loading data from npy format
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')

    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    y_true = np.load('data/y_true.npy')


    ###############################
    ##         Training          ##
    ###############################

    ##Changing to LSTM format
    X_train, y_train = prepare_data(X_train, y_train, mode='LSTM')
    X_val, y_val = prepare_data(X_val, y_val, mode='LSTM')
    X_test, y_test = prepare_data(X_test, y_test, mode='LSTM')


    #LSTM = LstmModel()
    #model, history = train(LSTM, X_train, y_train, X_val, y_val, save=True, save_model_path='./models/LSTM_n02.pth', save_history_path='./models/LSTM_n02_history.pkl')


    # X_train = X_train.transpose(1,2)
    # X_val = X_val.transpose(1,2)
    
    # CNN = CnnModel()
    # model, history = train(CNN, X_train, y_train, X_val, y_val, save=True, save_model_path='./models/CNN_n02.pth', save_history_path='./models/CNN_n02_history.pkl')

    #print(y_val.shape)

    # X_train = X_train.view(X_train.shape[0], -1)
    # X_val = X_val.view(X_val.shape[0], -1)

    # Forward = FeedForwardNet()
    # model, history = train(Forward, X_train, y_train, X_val, y_val, save=True, save_model_path='./models/Feed_n02.pth', save_history_path='./models/Feed_n02_history.pkl')



    ###############################
    ##        Predicting         ##
    ###############################
    day = 100

    day_1 = X_test[24*day]
    day_2 = X_test[24*(day+1)]
    
    observed = day_1[:,0]
    real_value = day_2[:,0]

    day_2 = day_2[:,1:] ##Removing target at column nb 0


    # model = LstmModel()  
    # model.load_state_dict(torch.load('./models/LSTM_n01.pth'), strict=False)
    # model.eval() 

    # predictions = predict_next_24_hours(model, day_1, day_2)

    # observed = denormalize(observed, './mean_values/normalization_LSTM_2.pkl')
    # real_value = denormalize(real_value, './mean_values/normalization_LSTM_2.pkl')
    # predictions = denormalize(predictions, './mean_values/normalization_LSTM_2.pkl')

    # plot_prediction(observed, real_value, predictions)


    # model = CnnModel()
    # model.load_state_dict(torch.load('./models/CNN_n02.pth'), strict=False)

    # predictions = predict_next_24_CNN(model, day_1, day_2)

    # observed = denormalize(observed, './mean_values/normalization_LSTM_2.pkl')
    # real_value = denormalize(real_value, './mean_values/normalization_LSTM_2.pkl')
    # predictions = denormalize(predictions, './mean_values/normalization_LSTM_2.pkl')


    # error = np.abs(np.array(real_value) - np.array(predictions))
    # np.save('./data/error_CNN.npy', error)




    # day_1 = day_1.view(day_1.shape[0], -1)
    # day_2 = day_2.view(day_2.shape[0], -1)

    # model = FeedForwardNet()
    # model.load_state_dict(torch.load('./models/Feed_n02.pth'), strict=False)

    # predictions = predict_next_24_feed(model, day_1, day_2)

    # observed = denormalize(observed, './mean_values/normalization_LSTM_2.pkl')
    # real_value = denormalize(real_value, './mean_values/normalization_LSTM_2.pkl')
    # predictions = denormalize(predictions, './mean_values/normalization_LSTM_2.pkl')

    # error = np.abs(np.array(real_value) - np.array(predictions))
    # np.save('./data/error_Feed.npy', error)





    ########################
    ###      Plots       ###
    ########################

    # Cnn_error = np.load('./data/error_CNN.npy')
    # Feed_error = np.load('./data/error_Feed.npy')
    # LSTM_error = np.load('./data/error_LSTM.npy')

    # plot_error(Cnn_error, Feed_error, LSTM_error)


    CNN_path = './models/CNN_n02_history.pkl'
    Feed_path = './models/Feed_n02_history.pkl'
    LSTM_path = './models/LSTM_n02_history.pkl'
    LSTM_path_2 = './models/LSTM_n01_history.pkl'

    with open(CNN_path, 'rb') as file:
        CNN_history = pickle.load(file) 
    
    with open(LSTM_path, 'rb') as file:
        LSTM_history = pickle.load(file) 
    
    with open(LSTM_path_2, 'rb') as file:
        LSTM_history_2 = pickle.load(file) 
    
    with open(Feed_path, 'rb') as file:
        Feed_history = pickle.load(file) 

    CNN_history = CNN_history['train']['mae']
    LSTM_history = LSTM_history['train']['mae']
    Feed_history = Feed_history['train']['mae']
    labels = ['CNN', 'LSTM', 'Feed Forward']

    print(CNN_history)

    plot_all_loss(CNN_history,LSTM_history, Feed_history, labels)

    

    