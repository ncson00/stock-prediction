import numpy as np
import pandas as pd
import pandas.io.sql as psql
import datetime
import argparse
import pickle
import yfinance as yf
from math import *
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

from db.postgres import *

lookback = 60


def preprocess(symbol, lookback):

    df = psql.read_sql(f"select date, close from stock_raw where stock = '{symbol}'", conn=connect())

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = df.filter(['Close']).values
    dataset = scaler.fit_transform(dataset)

    dataX, dataY = [], []
    for i in range(len(dataset) - lookback):
        row = [a for a in dataset[i:i+lookback]]
        dataX.append(row)
        dataY.append(dataset[i + lookback][0])
    X, y = np.array(dataX), np.array(dataY)

    # Train-test split
    split_point = int(len(dataset)*0.9)
    X_train, y_train = X[:split_point], y[:split_point]
    X_val, y_val = X[split_point:len(X)], y[split_point:len(y)]

    return scaler, X_train, y_train, X_val, y_val



def train(symbol, **kwargs):

    '''
    '''

    scaler, X_train, y_train, X_val, y_val = preprocess(symbol, lookback=60)

    ## Modeling
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Checkpoint
    cp = ModelCheckpoint(f'checkpoint/{symbol}/', save_best_only=True)

    # Compile the model
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[cp])

    # Save
    pickle.dump(model, open(f'model/LTSM_{symbol}.sav', 'wb'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Adding argument
    parser.add_argument("-s", "--symbol", help="")

    # Read arguments from command line
    args = parser.parse_args()
    if not args.symbol:
        raise IOError("Stock symbol must be specify from arguments!!!")

    train(args.symbol)
