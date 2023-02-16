import pandas as pd
import yfinance as yf
import datetime

from db.postgres import *

TICKETS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime("%Y-%m-%d")


def load_data():

    '''
    Get stock's historical data using yfinance and load into Postgres
    '''

    temp = pd.DataFrame()

    for ticket in TICKETS:
        try:
            data = yf.download(ticket, start=START_DATE, end=END_DATE, progress=False)
            data['date'] = pd.to_datetime(data.index)

            df = data.reset_index(drop=True)
            df.columns = df.columns.str.lower()
            df.columns = df.columns.str.replace(' ', '_')

            df['stock'] = ticket

            temp = pd.append([temp, df])

        except Exception as e:
            print("ERROR: Loading data failed!!! - {} - {}".format(ticket, e))

    return temp


if __name__ == '__main__':

    postgres_operator(
        query=f"delete from stock_raw;",
        conn=connect()
    )

    write_postgres(conn=connect(), df=load_data(), table_name='stock_raw')