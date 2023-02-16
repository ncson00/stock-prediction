import pandas as pd
import yfinance as yf

from db.postgres import *

TICKETS = ['AAPL']
START_DATE = '2015-01-01'
END_DATE = '2023-02-01'


def load_data():

    '''
    '''
    temp = pd.DataFrame()

    for s in TICKETS:
        try:
            data = yf.download(s, start=START_DATE, end=END_DATE, progress=False)
            data['date'] = pd.to_datetime(data.index)

            df = data.reset_index(drop=True)
            df.columns = df.columns.str.lower()
            df.columns = df.columns.str.replace(' ', '_')

            df['stock'] = s

            temp = pd.append([temp, df])

        except Exception as e:
            print("ERROR: Loading data failed!!! - {} - {}".format(s, e))

    return temp


if __name__ == '__main__':

    write_postgres(conn=connect(), df=load_data(), table_name='stock_raw')