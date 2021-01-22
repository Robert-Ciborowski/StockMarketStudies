import json
from datetime import datetime
import pandas as pd
import requests

from utils import getKeys

api_key = getKeys()["alpha_vantage_key"]

def downloadStockData(ticker: str):
    print("Starting download of", ticker, "...")
    df = None

    for year in range(2, 0, -1):
        for month in range(12, 0, -1):
            slice = "year" + str(year) + "month" + str(month)
            print("Downloading", ticker, "for", slice)
            data = downloadStockDataChunk(ticker, slice)

            if df is None:
                df = data
            else:
                df = df.append(data)

    return df

def downloadStockDataChunk(ticker: str, slice: str):
    endpoint = "https://www.alphavantage.co/query"

    params = {
        "function": "TIME_SERIES_INTRADAY_EXTENDED",
        "symbol": ticker,
        "interval": "1min",
        "slice": slice,
        "adjusted": "false",
        "apikey": api_key
    }

    page = requests.get(url=endpoint, params=params)
    content = str(page.content)[2 : len(page.content) - 1]
    content = content.split("\\r\\n")
    list_of_data = []

    for i in range(1, len(content)):
        data = content[i].split(",")

        if len(data) < 6:
            continue
        try:
            data[0] = pd.to_datetime(datetime.strptime(data[0], "%Y-%m-%d %H:%M:%S"))
            data[1] = float(data[1])
            data[2] = float(data[2])
            data[3] = float(data[3])
            data[4] = float(data[4])
            data[5] = int(data[5])
            list_of_data.append(data)
        except:
            continue

    df = pd.DataFrame(data=list_of_data[::-1], columns=content[0].split(","))
    return df

def main():
    data = downloadStockData("SPY")
    data.to_csv("spy_study/spy.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
