from utils import getKeys
from questrade_api import Questrade
from datetime import datetime

questrade_token = getKeys()["questrade_token"]

def main():
    # If using a new token:
    # q = Questrade(refresh_token=questrade_token)
    # otherwise...
    q = Questrade()

    # First, we need to find the ID of the symbol.
    id = q.symbols_search(prefix="SPY")["symbols"][0]["symbolId"]

    # Then we can get some candles.
    startTime = datetime(year=2020, month=10, day=2, hour=12, minute=0).astimezone().isoformat('T')
    endTime = datetime(year=2020, month=10, day=2, hour=12, minute=10).astimezone().isoformat('T')
    candles = q.markets_candles(id, interval='OneMinute', startTime=startTime,
                                endTime=endTime)
    print("Done!")


if __name__ == '__main__':
    main()
