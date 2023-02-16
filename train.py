import numpy as np
import pandas as pd
import pandas.io.sql as psql
import datetime
import argparse
import pickle
from math import *
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

from db.postgres import *


class DataHandler:

    '''
    '''

    def __init__(self, ticket, lookback = 60):
        self.ticket = ticket
        self.lookback = lookback


    def get_scaler(self):
        df = psql.read_sql(f"select date, close from stock_raw where stock = '{self.ticket}'", connect())
        dataset = df[['close']].values

        scaler = MinMaxScaler(feature_range=(0,1))
        dataset = scaler.fit_transform(dataset)

        self.dataset = dataset

        return scaler


    def model_input(self):

        self.get_scaler()

        dataX, dataY = [], []
        for i in range(len(self.dataset) - self.lookback):
            row = [a for a in self.dataset[i:i+self.lookback]]
            dataX.append(row)
            dataY.append(self.dataset[i + self.lookback][0])

        X, y = np.array(dataX), np.array(dataY)

        # Train-test split
        split_point = int(len(self.dataset)*0.9)
        X_train, y_train = X[:split_point], y[:split_point]
        X_val, y_val = X[split_point:len(X)], y[split_point:len(y)]

        return X_train, y_train, X_val, y_val


def baseline_model(ticket):

    '''
    '''
    model = DataHandler(ticket=ticket, lookback=60)

    X_train, y_train, X_val, y_val = model.model_input()

    ### Modeling
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Checkpoint
    cp = ModelCheckpoint(f'checkpoint/{ticket}/', save_best_only=True)

    # Compile the model
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[cp])

    # Save
    pickle.dump(model, open(f'model/LTSM_{ticket}.sav', 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Adding argument
    parser.add_argument("-s", "--ticket", help="")

    # Read arguments from command line
    args = parser.parse_args()
    if not args.ticket:
        raise IOError("Stock ticket must be specify from arguments!!!")

    baseline_model(args.ticket)