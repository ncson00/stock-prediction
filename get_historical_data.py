import pymongo
from pymongo import MongoClient, UpdateOne
import json
import pandas as pd
from datetime import datetime
import time
import requests
import pprint

MONGO_URI = 'mongodb://localhost:27017/'

API_KEY = '8816ce5d529c472e9873fbb23165f0c1'
INTERVAL = '1day'
SYMBOL = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'MA', 'BRK.A' 'TLSA', 'NVDA', 'V', 'XOM']
# SYMBOL = ['AAPL']
START_DATE = '2014-12-31 00:00:00'
END_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def store_data(_symbol):

    '''
    '''
    try:
        client = MongoClient(MONGO_URI)
        db = client.stock
        db[_symbol].drop()
        collection = db[_symbol]

        try: 
            response = requests.get(f"https://api.twelvedata.com/time_series?apikey={API_KEY}" + \
                                                                            f"&interval={INTERVAL}" + \
                                                                            f"&symbol={_symbol}" + \
                                                                            f"&country=US" + \
                                                                            f"&timezone=exchange" + \
                                                                            f"&start_date={START_DATE}&end_date={END_DATE}" + \
                                                                            f"&format=JSON")
            # print(response.text)
            # print(50*'~')

            historical_price = response.json()
            historical_price = historical_price['values']
            
            # for row in historical_price:
            #     row['_id'] = row.pop('datetime')

            # collection.insert_many(historical_price, ordered=False)

            gen_request = lambda x: UpdateOne(
                {'_id': x['datetime']},
                {'$set': x},
                upsert=True
            )
            temp = list(map(gen_request, historical_price))

            collection.bulk_write(temp)
            
        except Exception as err:
            print('ERROR: ', err)
            return False

        client.close()

    except Exception as e:
        print("ERROR: Connection failed - ", e)
        return False


if __name__ == '__main__':

    '''
    '''
    for symbol in SYMBOL:
        store_data(_symbol=symbol)
        time.sleep(10)