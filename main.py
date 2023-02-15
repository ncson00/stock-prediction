import numpy as np
import pandas as pd
import pandas.io.sql as psql
from datetime import datetime, timedelta
import argparse
import psycopg2
import psycopg2.extras as extras
import pickle

from math import *
from sklearn.metrics import mean_squared_error

from train import DataHandler
from db.postgres import *

PG_USER = "postgres"
PG_PASSWORD = "1"
PG_DATABASE = "stock"
PG_HOST = "172.18.0.3"
PG_PORT = 5432
PG_DRIVER = 'org.postgresql.Driver'

start_date = '2015-01-01'
end_date = '2023-02-01'


def load_model(symbol):
    return pickle.load(open(f'model/LTSM_{symbol}.sav', 'rb'))


def update_rmse(symbol, run_date):

    '''
    '''
    model = DataHandler(symbol=symbol, lookback=60)

    X_train, y_train, X_val, y_val = model.preprocess()

    ltsm = load_model(symbol)
    y_predict = ltsm.predict(X_val)

    scaler = model.get_scaler()
    predict_price = scaler.inverse_transform(y_predict.reshape(-1, 1))
    true_price = scaler.inverse_transform(y_val.reshape(-1, 1))

    rmse = sqrt(mean_squared_error(true_price, predict_price))

    postgres_operator(
        query=f"""
            delete from rmse where date = '{run_date}';
            insert into rmse values ({run_date}, {symbol}, {rmse});
        """,
        conn=connect()
    )


def recursive_forecasting(symbol, run_date, steps):

    '''
    '''
    # Load input
    data = psql.read_sql(f"""
        select date, close 
        from stock_raw 
        where stock = '{symbol}'
            and date between '{run_date - timedelta(days=90)}' and '{run_date}'
    """, connect())

    model = DataHandler()
    scaler = model.get_scaler()
    input = scaler.transform(data[['close']].values)

    ltsm = load_model(symbol)

    for step in steps:
        temp = ltsm.predict(input)
        input = np.append(input, temp[-1])