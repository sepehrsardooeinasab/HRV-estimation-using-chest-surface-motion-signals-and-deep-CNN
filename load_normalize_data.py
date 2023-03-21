from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def load_data(address, trial_names, Y_name):
    X = list()
    
    search_values = [i + '_' for i in trial_names]
    hrv = pd.read_csv(address + '/HRV.csv')
    hrv = hrv.set_index('Trial Name')
    hrv.index.name = None
    
    for index in hrv[hrv.index.str.contains('|'.join(search_values))].index:
        motions_spectogram = np.load(address + '/' + index + '.npy')
        X.append(motions_spectogram)
    
    X = np.stack(X, axis=0)
    Y = hrv[Y_name][hrv.index.str.contains('|'.join(search_values))]
    Y = Y.to_numpy()
    
    return X, Y


def normalize(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_val, X_test