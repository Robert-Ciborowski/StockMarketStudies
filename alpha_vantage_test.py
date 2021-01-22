from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

from utils import getKeys

api_key = getKeys()["alpha_vantage_key"]

def getIntraDay(ticker: str, interval="1min", full=False):
    """
    Gets intra day data.
    :param ticker: e.g. TSLA
    :param interval: 1min, 5min, 15min, 30min, 60min
    :return: Pandas Dataframe
    """
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=ticker, interval=interval,
                                      outputsize='full')
    data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low",
                 "4. close": "Close", "5. volume": "Volume"})
    return data

def main():
    data = getIntraDay("SPY")
    data["Close"].plot()
    plt.title("Intraday Times Series for TSLA (1 min)")
    plt.show()

if __name__ == "__main__":
    main()
