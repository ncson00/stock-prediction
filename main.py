import numpy as np
import pandas as pd
import pandas.io.sql as psql
from datetime import datetime, timedelta, date
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

        X_train, y_train, X_val, y_val = data_handler.model_input()

        model = self.load_model()
        y_predict = model.predict(X_val)

        scaler = data_handler.get_scaler()
        predict_price = scaler.inverse_transform(y_predict.reshape(-1, 1))
        true_price = scaler.inverse_transform(y_val.reshape(-1, 1))

        rmse = sqrt(mean_squared_error(true_price, predict_price))
        print("RMSE: {}".format(rmse))
        return rmse


    def recursive_forecasting(self):

        '''
        '''
        # Load input
        data = psql.read_sql(f"""
            select close 
            from stock_raw 
            where stock = '{self.ticket}'
                and date between '{self.run_date - timedelta(days=150)}' and '{self.run_date}'
        """, connect())

        true_price = data['close'].values[-1]

        data_handler = DataHandler(ticket=self.ticket, lookback=60)
        scaler = data_handler.get_scaler()
        scaled_data = scaler.transform(data[['close']].values)

        # Reshape scaled data
        input = np.reshape(scaled_data, len(scaled_data))


        model = self.load_model()

        for step in range(5):
            temp = model.predict(data_handler.transform_array(input))
            input = np.append(input, temp[-1])

        result = scaler.inverse_transform(input.reshape(-1, 1))
        output = np.reshape(result, len(result))

        return output, true_price
    

def main(run_date, ticket):

    '''
    '''
    model_output = ModelOutput(run_date=run_date, ticket=ticket, lookback=60)
    rmse = model_output.load_rmse()
    print(rmse)
    output, true_price = model_output.recursive_forecasting()
    print(output)
    print(true_price)

    postgres_operator(
        query=f"""
            delete from predict_result where date = '{run_date}';
            insert into predict_result values ('{run_date}', '{ticket}', {true_price}, {output[-5]}, {output[-3]}, {output[-1]}, {rmse});
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
        args.run_date = date.today()
    if not args.ticket:
        raise IOError("Ticket must be specify from arguments!!!")

    main(args.run_date, args.ticket)