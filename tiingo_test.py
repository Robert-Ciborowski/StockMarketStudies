import requests

from utils import getKeys

tiingo_token = getKeys()["tiingo_token"]

def main():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Token " + tiingo_token
    }
    requestResponse = requests.get("https://api.tiingo.com/iex/SPY/prices?"\
            "startDate=2016-12-17&endDate=2016-12-20&resampleFreq=5min",
                                   headers=headers)
    print(requestResponse.json())

if __name__ == "__main__":
    main()
