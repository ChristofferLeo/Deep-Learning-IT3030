import pandas as pd
import pickle
import json
import numpy as np

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

def normalize(data, save_path='mean_values/normalization_LSTM_2.pkl'):
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
    y_true =[]
    for i in range(len(data) - seq_length - 1):
        # Original features for the sequence
        x_features = data.iloc[i:(i + seq_length), 1:]  
        
        # Shifted target (y at t-1) to be included as part of the features
        shifted_y = data.iloc[i:(i + seq_length), 0].shift(1)
        
        # Combine the features with the shifted_y
        x_combined = pd.concat([shifted_y, x_features], axis=1).iloc[1:,]  
        
        # True target value for the sequence is the single next value following the sequence
        y = data.iloc[i + seq_length, 0]

        #True target unsqueezed
        true_sequence = data.iloc[(i+1):(i+1+seq_length), 0]
        
        xs.append(x_combined)
        ys.append(y)  
        y_true.append(true_sequence)

    # Convert the list of pandas DataFrames and values into arrays
    xs = np.array([x.values for x in xs])
    ys = np.array(ys)  # ys is already suitable as a list of single values; just convert to numpy array
    
    return xs, ys, y_true
    
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


def preprocessing(fileName, area, seq_length= 24):
    print("Starting preprocessing data....")

    # Reading from file
    data = read_data(fileName, area)
    print("Data read from file sucsessfully")

    # Getting time on right format
    data = split_timestamp(data)

    #Normalizing the data
    data = normalize(data)
    print("Data normalized")

    data = data.rename(columns={area[1]: 'target'})

    X, y, y_true = create_sequences(data, seq_length)
    
    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print("Data splitted")

    # Save X_train and y_train as .npy files
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)

    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)

    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    print("Preprocessing data complete!")


## Features dependign on areas
area_1 = ['timestamp', 'NO1_consumption', 'NO1_temperature']
area_2 = ['timestamp', 'NO2_consumption', 'NO2_temperature']

fileName = 'test_set.csv'



##Change this if other area is prefered
preprocessing(fileName, area_1)