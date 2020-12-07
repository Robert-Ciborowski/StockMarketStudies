from typing import List

import requests
import json

from utils import getKeys

td_consumer_key = getKeys()["td_consumer_key"]

def getCurrentQuotes(tickers: List):
    params = {
        "apikey": td_consumer_key,
        "symbol": tickers
    }

    endpoint = "https://api.tdameritrade.com/v1/marketdata/quotes"
    page = requests.get(url=endpoint, params=params)
    content = json.loads(page.content)
    return content

def getPriceHistory(ticker: str, periodType="day", period=1, frequency=1,
                    frequencyType="minute"):
    """
    Returns price history in candles (open, high, low, close).

    To get back candles for the last two days, and get candles
    for every minute: periodType="day", period=2, frequency=1,
    frequencyType="minute".

    Valid periods by periodType (according to the TD API):

    day: 1, 2, 3, 4, 5, 10
    month: 1, 2, 3, 6
    year: 1, 2, 3, 5, 10, 15, 20
    ytd: 1

    Valid frequencies by frequencyType (according to the TD API):

    minute: 1, 5, 10, 15, 30
    daily: 1
    weekly: 1
    monthly: 1py

    :param ticker: e.g. TSLA
    :param periodType: day, month, year, ytd
    :param period: number of periods to get back, integer
    :param frequency: integer
    :param frequencyType: minute, daily, weekly, monthly
    :return:
    """
    endpoint = "https://api.tdameritrade.com/v1/marketdata/" + ticker + "/pricehistory"

    params = {
        "apikey": td_consumer_key,
        "periodType": periodType,
        "period": str(period),
        "frequency": str(frequency),
        "frequencyType": frequencyType
    }

    page = requests.get(url=endpoint, params=params)
    content = json.loads(page.content)
    return content


def main():
    # print(getCurrentQuotes(["AAPL", "TSLA"]))
    print(getPriceHistory("TSLA"))

if __name__ == '__main__':
    main()
