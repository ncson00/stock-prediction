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


class ModelOutput(DataHandler):

    '''
    '''
    def __init__(self, run_date, ticket, lookback):
        super().__init__(ticket, lookback)
        self.run_date = run_date


    def load_model(self):
        return pickle.load(open(f'model/LTSM_{self.ticket}.sav', 'rb'))


    def update_rmse(self):

        model = DataHandler(ticket=self.ticket, lookback=60)

        X_train, y_train, X_val, y_val = model.preprocess()

        ltsm = self.load_model()
        y_predict = ltsm.predict(X_val)

        scaler = model.get_scaler()
        predict_price = scaler.inverse_transform(y_predict.reshape(-1, 1))
        true_price = scaler.inverse_transform(y_val.reshape(-1, 1))

        rmse = sqrt(mean_squared_error(true_price, predict_price))

        return rmse


    def recursive_forecasting(ticket, run_date, steps):

        '''
        '''
        # Load input
        data = psql.read_sql(f"""
            select close 
            from stock_raw 
            where stock = '{ticket}'
                and date between '{run_date - timedelta(days=90)}' and '{run_date}'
        """, connect())

        model = DataHandler()
        scaler = model.get_scaler()
        input = scaler.transform(data[['close']].values)

        ltsm = load_model(ticket)

        for steps in [1, 3, 5]:
            for step in range(steps):
                temp = ltsm.predict(input)
                input = np.append(input, temp[-1])

            result = scaler.inverse_transform(input.reshape(-1, 1))[-1]


        postgres_operator(
            query=f"""
                
            """
        )