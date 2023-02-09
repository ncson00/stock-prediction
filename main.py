from pymongo import MongoClient
import json
import pandas as pd
import numpy as np
import requests
import argparse

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

MONGO_URI = 'mongodb://localhost:27017/'
lookback = 60


def list_to_dataframe(df):

    df = df.drop('_id', axis=1)

    for col in df.columns:
        if col == 'datetime': 
            df[col] = pd.to_datetime(df[col], format= '%Y-%m-%d')
        else:
            df[col] = df[col].astype(float)

    return df.sort_values(['datetime'], ascending=True)


def read_mongo(_symbol):

    '''
    '''
    try:
        client = MongoClient(MONGO_URI)
        db = client.stock
        collection = db[_symbol]

        list_cursor = list(collection.find())

        to_df = pd.DataFrame(list_cursor)
        df = list_to_dataframe(to_df)
    
        print('Connect MongoDB successfully!')

    except Exception as e:
        print('ERROR: ', e)
        return

    return df


# Create input dataset
def input_dataset(dataset, lookback):

    dataX, dataY = [], []

    for i in range(len(dataset) - lookback):
        row = [a for a in dataset[i:i+lookback]]
        dataX.append(row)
        dataY.append(dataset[i + lookback][0])

    return np.array(dataX), np.array(dataY)


# Convert array to input for recursive forecastng
def array_to_input(array, lookback):

    dataX = []

    for i in range(len(array) - lookback + 1):
        row = [[a] for a in array[i:i+lookback]]
        dataX.append(row)

    return np.array(dataX)


def recursive_forecasting(model, scaler, input, list_timestamp, step, **kwargs):

    for i in range(step):
        tmp = model.predict(array_to_input(input, lookback))
        input = np.append(input, tmp[-1])
    
    while len(list_timestamp) < len(input):
        list_timestamp = np.append(list_timestamp, list_timestamp[-1] + np.timedelta64(1, 'D'))
    
    output_predict = scaler.inverse_transform(input.reshape(-1, 1))
    
    return pd.DataFrame({
        'timestamp': list_timestamp,
        'predict': np.reshape(output_predict, len(output_predict))
    })
    


def main(symbol, **kwargs):

    '''
    '''
    # Create a new dataframe with only the 'Close column 
    df = read_mongo(_symbol=symbol)
    data = df.filter(['close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    print(len(dataset))
    print(50*'~')
    # Get the number of rows to train the model on
    split_point = int(len(dataset)*85/100)

    # Min Max Scale
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(dataset)

    # Input dataset
    X, y = input_dataset(scaled, lookback)

    print('Data train shape: {}'.format(X.shape))
    print('Data test shape: {}'.format(y.shape))
    print(50*'~')

    # Train-test split
    X_train, y_train = X[:split_point], y[:split_point]
    X_val, y_val = X[split_point:len(X)], y[split_point:len(y)]


    ## Modeling
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Checkpoint
    cp = ModelCheckpoint('forescasting/', save_best_only=True)

    # Compile the model
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    # Train the model
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), callbacks=[cp])


    ## Result
    y_predict = model.predict(X_val)

    predictY_inverse = scaler.inverse_transform(y_predict.reshape(-1, 1))
    valY_inverse = scaler.inverse_transform(y_val.reshape(-1, 1))

    result = pd.DataFrame({
        'timestamp': df.datetime[-len(y_val):],
        'true': np.reshape(valY_inverse, len(valY_inverse)),
        'predict': np.reshape(predictY_inverse, len(predictY_inverse)),
    })

    rmse = sqrt(mean_squared_error(valY_inverse, predictY_inverse))
    print('Test RMSE: %.3f' % rmse)
    print(50*'~')


    forecast = recursive_forecasting(
        model = model,
        scaler = scaler,
        input = y_val, 
        list_timestamp = df.datetime[-len(y_val):].values, 
        step = 1
    )
    print(forecast.tail(10))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Adding argument
    parser.add_argument("-s", "--symbol", help="")

    # Read arguments from command line
    args = parser.parse_args()
    if not args.symbol:
        raise IOError("Stock symbol must be specify from arguments!!!")

    main(args.symbol)
