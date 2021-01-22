from spy_study.Constants import TRADING_MINUTES_TO_ANALYSE
from spy_study.Model import Model
import pandas as pd
import csv
import numpy as np

def read_csv(path: str):
    features = []
    labels = []
    tradingMinutesPerDay = TRADING_MINUTES_TO_ANALYSE

    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first = False

        for row in reader:
            if not first:
                first = True
                continue

            prices = np.asarray([float(x) for x in row[0 : tradingMinutesPerDay]])
            volumes = np.asarray([float(x) for x in row[tradingMinutesPerDay : tradingMinutesPerDay * 2]])
            rsi_day = np.asarray([float(x) for x in row[tradingMinutesPerDay * 2 : tradingMinutesPerDay * 3]])
            rsi_hour = np.asarray([float(x) for x in row[tradingMinutesPerDay * 3 : tradingMinutesPerDay * 4]])
            feature = np.asarray([prices, volumes, rsi_day, rsi_hour]).T
            features.append(feature)
            labels.append(np.asarray([float(row[len(row)-1])]))

    return np.asarray(features), np.asarray(labels)


def main():
    features, labels = read_csv("dataset.txt")
    training_features = features[:400]
    training_labels = labels[:400]
    testing_features = features[400:]
    testing_labels = labels[400:]

    epoch = 4
    learningRate = 0.01
    batchSize = 10
    validationSplit = 0.1

    model = Model()
    model.setup()
    model.createModel(learningRate)
    model.trainModel(training_features, training_labels, validationSplit, batchSize, epoch)
    # model.exportWeights()

    model.loadWeights()

    # Simulate model on its own dataset
    averageAbsLoss, predictions = model.simulatePerformanceOnTrades(testing_features, testing_labels)
    print("Average Absolute Loss on test set:", averageAbsLoss)

    # Predict the price of Dec 15, 2020
    predictedPrice = model.makePricePrediction(testing_features[-2], 364.63)
    print("Predicted price for Dec 15, 2020:", predictedPrice)
    print("Done!")

    # If the model guesses whether the price will be up or down, how many
    # times will it be correct?
    percentCorrect = model.simulateUpDownPerformance(testing_features, testing_labels)
    print("Percent of up/down guesses correct on test set:", percentCorrect)

    # data = pd.read_csv("spy.csv")
    # data["time"] = pd.to_datetime(data["time"])

if __name__ == '__main__':
    main()
