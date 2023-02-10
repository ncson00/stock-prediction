import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import psycopg2
import psycopg2.extras as extras
import pickle

from math import *
from sklearn.metrics import mean_squared_error

from train import preprocess
from db.postgres import *

PG_USER = "postgres"
PG_PASSWORD = "1"
PG_DATABASE = "stock"
PG_HOST = "172.18.0.3"
PG_PORT = 5432
PG_DRIVER = 'org.postgresql.Driver'

start_date = '2015-01-01'
end_date = '2023-02-01'


def update_rmse(symbol, run_date):

    '''
    '''
    model = pickle.load(open(f'model/LTSM_{symbol}.sav', 'rb'))
    scaler, X_train, y_train, X_val, y_val = preprocess(symbol, lookback=60)


    y_predict = model.predict(X_val)

    predict_price = scaler.inverse_transform(y_predict.reshape(-1, 1))
    true_price = scaler.inverse_transform(y_val.reshape(-1, 1))

    rmse = sqrt(mean_squared_error(true_price, predict_price))

    postgres_oprerator(
        sql=f"""
            delete from rmse where date = '{run_date}';
            insert into rmse values ({run_date}, {symbol}, {rmse});
        """,
        conn=connect()
    )


