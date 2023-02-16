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


class ModelOutput(DataHandler):

    '''
    '''
    def __init__(self, run_date, ticket, lookback):
        super().__init__(ticket, lookback)
        self.run_date = run_date


    def load_model(self):
        return pickle.load(open(f'model/LTSM_{self.ticket}.sav', 'rb'))


    def load_rmse(self):

        '''Calculating Model's RMSE
        '''

        data_handler = DataHandler(ticket=self.ticket, lookback=60)

        X_train, y_train, X_val, y_val = data_handler.preprocess()

        model = self.load_model()
        y_predict = model.predict(X_val)

        scaler = data_handler.get_scaler()
        predict_price = scaler.inverse_transform(y_predict.reshape(-1, 1))
        true_price = scaler.inverse_transform(y_val.reshape(-1, 1))

        rmse = sqrt(mean_squared_error(true_price, predict_price))
        return rmse


    def recursive_forecasting(self):

        '''
        '''
        # Load input
        data = psql.read_sql(f"""
            select close 
            from stock_raw 
            where stock = '{self.ticket}'
                and date between '{self.run_date - timedelta(days=90)}' and '{self.run_date}'
        """, connect())

        data_handler = DataHandler()
        scaler = data_handler.get_scaler()
        input = scaler.transform(data[['close']].values)
        true_price = data['close'][-1]

        model = self.load_model()

        output = []
        for steps in [1, 3, 5]:
            for step in range(steps):
                temp = model.predict(input)
                input = np.append(input, temp[-1])

            result = scaler.inverse_transform(input.reshape(-1, 1))
            output = np.append(output, result[-1])

        return output, true_price
    

def main(run_date, ticket):

    '''
    '''
    model_output = ModelOutput(run_date=run_date, ticket=ticket, lookback=60)
    rmse = model_output.load_rmse()
    output, true_price = model_output.recursive_forecasting()

    postgres_operator(
        query=f"""
            delete from predict_result where date = '{run_date}';
            insert into predict_result values ({run_date}, {ticket}, {true_price}, {output[0]}, {output[1]}, {output[2]}, {rmse});
        """, conn=connect()
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Adding argument
    parser.add_argument("-d", "--run_date", help="")
    parser.add_argument("-t", "--ticket", help="")

    # Read arguments from command line
    args = parser.parse_args()
    if not args.run_date:
        raise IOError("Run date must be specify from arguments!!!")
    if not args.ticket:
        raise IOError("Ticket must be specify from arguments!!!")

    main(args.run_date, args.ticket)