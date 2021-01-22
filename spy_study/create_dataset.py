import math
import pandas as pd
import stock_pandas as spd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from spy_study.Constants import MAX_ALLOWED_CHANGE_FOR_DATASET, \
    MIN_ALLOWED_CHANGE_FOR_DATASET, TRADING_MINUTES_TO_ANALYSE


def plot(df: pd.DataFrame, startTime: datetime, endTime: datetime):
    # This sets up the data.
    mask = (df["time"] > pd.to_datetime(startTime)) & (df["time"] <= pd.to_datetime(endTime))
    dataToPlot = df.loc[mask]

    # This plots!
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    dataToPlot.plot(ax=axes[0], x="time", y="close", label="Close", color="blue")
    dataToPlot.plot(ax=axes[1], x="time", y="volume", label="Volume", color="green")
    dataToPlot.plot(ax=axes[2], x="time", y="rsi_day", label="RSI - Day", color="purple")
    dataToPlot.plot(ax=axes[3], x="time", y="rsi_hour", label="RSI - Hour", color="purple")
    axes[0].set_title("Close Price")
    axes[1].set_title("Volume")
    axes[2].set_title("RSI - Day")
    axes[3].set_title("RSI - Hour")
    axes[2].set_ylim([0, 100])
    axes[3].set_ylim([0, 100])
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle(str(startTime) + " to " + str(endTime))
    fig.show()

def addDayRSI(data: pd.DataFrame):
    # This turns our data into stock-pandas daily data (for RSI calculation)
    daily_data = data[["time", "close"]]
    daily_data["datetime_2"] = daily_data["time"].apply(
        lambda x: x.replace(hour=0, second=0, minute=0))
    daily_data = daily_data.drop_duplicates(subset="datetime_2", keep="last")
    daily_data = daily_data.drop(columns=["time"])
    daily_data = daily_data.rename(columns={"datetime_2": "time"})
    daily_data = daily_data.reset_index()
    daily_data = spd.StockDataFrame(daily_data)

    # Add daily RSI. Note that the typical period for stocks is 14 days.
    rsi_to_use = "rsi:14"
    daily_data[rsi_to_use]

    # Construct our final dataset
    def getRSI(dt: datetime):
        position = daily_data.index[
            daily_data["time"] == dt.replace(hour=0, minute=0,
                                                 second=0)].tolist()

        if len(position) == 0:
            return 0.0

        position = position[0]
        row = daily_data.iloc[position]
        rsi = row[rsi_to_use]

        if math.isnan(rsi):
            return 0.0
        else:
            return rsi

    data["rsi_day"] = data.apply(lambda x: getRSI(x["time"]), axis=1)

def addHourRSI(data: pd.DataFrame):
    # This turns our data into stock-pandas daily data (for RSI calculation)
    daily_data = data[["time", "close"]]
    daily_data["datetime_2"] = daily_data["time"].apply(
        lambda x: x.replace(second=0, minute=0))
    daily_data = daily_data.drop_duplicates(subset="datetime_2", keep="last")
    daily_data = daily_data.drop(columns=["time"])
    daily_data = daily_data.rename(columns={"datetime_2": "time"})
    daily_data = daily_data.reset_index()
    daily_data = spd.StockDataFrame(daily_data)

    # Add daily RSI. Note that the typical period for stocks is 14 days.
    rsi_to_use = "rsi:14"
    daily_data[rsi_to_use]

    # Construct our final dataset
    def getRSI(dt: datetime):
        position = daily_data.index[
            daily_data["time"] == dt.replace(minute=0,
                                                 second=0)].tolist()

        if len(position) == 0:
            return 0.0

        position = position[0]
        row = daily_data.iloc[position]
        rsi = row[rsi_to_use]

        if math.isnan(rsi):
            return 0.0
        else:
            return rsi

    data["rsi_hour"] = data.apply(lambda x: getRSI(x["time"]), axis=1)

def findCloseToOpenChanges(data: pd.DataFrame):
    datetimesForDataset = []
    percentChangeNextDay = []

    first_date = data.iloc[0]["time"]
    first_date = first_date.replace(hour=15, minute=45, second=0)
    last_date = data.tail(n=1).iloc[0]["time"]
    last_date = last_date.replace(hour=10, minute=0, second=0)

    while first_date < last_date:
        # Next day, 10:00 am:
        next_day_open_date = first_date + timedelta(hours=17, minutes=45)
        next_day_close_date = first_date + timedelta(days=1)
        current_day_close = data.loc[data["time"] == first_date]

        if current_day_close.shape[0] == 0:
            first_date += timedelta(days=1)
            continue

        current_day_close = current_day_close.iloc[0]
        condition1 = data["time"] > next_day_open_date
        condition2 = data["time"] < next_day_close_date
        next_day_open = data.loc[condition1 & condition2]

        if next_day_open.shape[0] == 0:
            # It might currently be Friday. Skip if it isn't.
            if first_date.weekday() != 4:
                first_date += timedelta(days=1)
                continue

            next_day_open_date = first_date + timedelta(days=2, hours=18, minutes=15)
            next_day_close_date = first_date + timedelta(days=3)
            condition1 = data["time"] > next_day_open_date
            condition2 = data["time"] < next_day_close_date
            next_day_open = data.loc[condition1 & condition2]

            if next_day_open.shape[0] == 0:
                # Give up.
                first_date += timedelta(days=1)
                continue

        # next_day_open = next_day_open.iloc[0]["close"]
        next_day_open = next_day_open["close"].max()

        datetimesForDataset.append(first_date)
        percentChangeNextDay.append(
            next_day_open / current_day_close["close"])
        print(first_date)
        first_date += timedelta(days=1)

    return datetimesForDataset, percentChangeNextDay

def createDataset(data: pd.DataFrame):
    maxAllowedChange = MAX_ALLOWED_CHANGE_FOR_DATASET
    minAllowedChange = MIN_ALLOWED_CHANGE_FOR_DATASET

    # Find changes between closes and opens. Also discounts
    # weird days, e.g. weekends or days when the market closed early.
    datetimesForDataset, percentChangeNextDay = findCloseToOpenChanges(data)

    # Number of trading minutes in a day: 6.25 hours * 60 minutes/hour
    # (9:30 - 15:45)
    tradingMinutes = TRADING_MINUTES_TO_ANALYSE
    datasetFeatures = []

    for i in range(len(datetimesForDataset)):
        todayClose = datetimesForDataset[i]
        todayOpen = todayClose - timedelta(minutes=tradingMinutes)
        # This sets up the data.
        mask = (data["time"] >= pd.to_datetime(todayOpen)) & (
                data["time"] < pd.to_datetime(todayClose))
        todayData = data.loc[mask]

        if len(todayData) != tradingMinutes:
            continue

        datasetFeatures.append(todayData)

        # Clamps the percent change between 0.8 & 1.2. Then it scales the final
        # value to be a number between 0 and 1.
        val = percentChangeNextDay[i]
        val = min(maxAllowedChange, max(minAllowedChange, val))
        diff = maxAllowedChange - minAllowedChange
        percentChangeNextDay[i] = val / diff - minAllowedChange / diff

    # Return the x and y for our dataset.
    return datasetFeatures, percentChangeNextDay

def writeDataset(datasetX, datasetY):
    data = ""

    # Pick a random entry to find the number of columns.
    number_of_columns = len(datasetX[1]["close"]) * 4 + 1

    for i in range(number_of_columns):
        data += str(i)

        if i != number_of_columns - 1:
            data += ","

    data += "\n"

    for i in range(len(datasetX)):
        x = datasetX[i]
        prices = (x["close"] - x["close"].mean()) / x["close"].std()
        prices = prices / prices.max()
        volumes = (x["volume"] - x["volume"].mean()) / x["volume"].std()
        volumes = volumes / volumes.max()
        rsi_day = x["rsi_day"] / 100
        rsi_hour = x["rsi_hour"] / 100

        for value in prices:
            data += str(value) + ","

        for value in volumes:
            data += str(value) + ","

        for value in rsi_day:
            data += str(value) + ","

        for value in rsi_hour:
            data += str(value) + ","

        data += str(datasetY[i]) + "\n"

    f = open("dataset.txt", "w")
    f.write(data)
    f.close()

def main():
    # This reads our data.
    data = pd.read_csv("spy.csv")
    data["time"] = pd.to_datetime(data["time"])

    addDayRSI(data)
    addHourRSI(data)

    # Plot the dataset
    # startDate = datetime(year=2020, month=11, day=2, hour=9, minute=30)
    startDate = datetime(year=2019, month=1, day=1, hour=0, minute=0)
    endDate = datetime(year=2020, month=12, day=14, hour=20, minute=0)
    # plot(data, startDate, endDate)

    # Create a dataset if it has not already been created.
    datasetX, datasetY = createDataset(data)
    writeDataset(datasetX, datasetY)

    print("Done.")

if __name__ == '__main__':
    main()
