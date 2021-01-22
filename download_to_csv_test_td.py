import requests
import json
from utils import getKeys
from datetime import datetime

td_consumer_key = getKeys()["td_consumer_key"]

epoch = datetime.utcfromtimestamp(0)

def unixTimeSinceMillis(dt):
    return (dt - epoch).total_seconds() * 1000.0

def getPriceHistory(ticker: str, startDate: datetime, endDate: datetime, frequency=1,
                    frequencyType="minute"):
    """
    Returns price history in candles (open, high, low, close).

    Valid frequencies by frequencyType (according to the TD API):

    minute: 1, 5, 10, 15, 30
    daily: 1
    weekly: 1
    monthly: 1py

    :param ticker: e.g. TSLA
    :param startDate:
    :param endDate:
    :param frequency: integer
    :param frequencyType: minute, daily, weekly, monthly
    :return:
    """
    endpoint = "https://api.tdameritrade.com/v1/marketdata/" + ticker + "/pricehistory"

    params = {
        "apikey": td_consumer_key,
        "endDate": str(int(unixTimeSinceMillis(endDate))),
        "startDate": str(int(unixTimeSinceMillis(startDate))),
        "frequency": str(frequency),
        "frequencyType": frequencyType
    }

    page = requests.get(url=endpoint, params=params)
    content = json.loads(page.content)
    return content

def main():
    # [startDate, endDate)
    startDate = datetime(year=2016, month=11, day=2, hour=7, minute=0)
    endDate = datetime(year=2020, month=12, day=11, hour=20, minute=1)
    history = getPriceHistory("SPY", startDate, endDate)
    history = history["candles"]
    data = "datetime,open,high,low,close,volume\n"

    for i in range(len(history)):
        timestamp = datetime.fromtimestamp(int(history[i]["time"] / 1000))
        data += str(timestamp) + "," + str(history[i]["open"])\
                + "," + str(history[i]["high"]) + "," + str(history[i]["low"])\
                + "," + str(history[i]["close"]) + "," + str(history[i]["volume"]) + "\n"

    f = open("spy_study/spy_small.csv", "w")
    f.write(data)
    f.close()

if __name__ == '__main__':
    main()
