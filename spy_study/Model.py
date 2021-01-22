import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers

from spy_study.Constants import MAX_ALLOWED_CHANGE_FOR_DATASET, \
    MIN_ALLOWED_CHANGE_FOR_DATASET, TRADING_MINUTES_TO_ANALYSE


class Model:
    listOfMetrics: List
    exportPath: str

    _metrics: List
    _MINUTES = TRADING_MINUTES_TO_ANALYSE
    _SAMPLES_OF_DATA = TRADING_MINUTES_TO_ANALYSE * 4

    def __init__(self, tryUsingGPU=False):
        super().__init__()

        if not tryUsingGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            self._configureForGPU()

        self.exportPath = "./model_exports/spy_model"

        # The following lines adjust the granularity of reporting.
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.1f}".format
        # tf.keras.backend.set_floatx('float32')

    def setup(self):
        self._buildMetrics()

    def detect(self, prices, volumes) -> float:
        pass

    """
    Creates a brand new neural network for this model.
    """
    def createModel(self, learningRate: float):
        # Should go over minutes, not seconds
        input_layer = layers.Input(shape=(self._MINUTES, 4))
        layer = layers.Conv1D(filters=16, kernel_size=16, activation='relu',
                         input_shape=(self._MINUTES, 4))(input_layer)
        layer = layers.AveragePooling1D(pool_size=2)(layer)
        layer = layers.Conv1D(filters=8, kernel_size=8, activation='relu',
                          input_shape=(self._MINUTES, 4))(layer)
        layer = layers.AveragePooling1D(pool_size=2)(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._MINUTES, input_shape=layer.shape))(layer)
        layer = layers.Dense(5, activation='relu')(layer)
        layer = layers.Dense(2, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
        layer = layers.Dense(1, activation='sigmoid')(layer)
        self.model = tf.keras.Model(input_layer, layer)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.RMSprop(lr=learningRate),
                           metrics=self._metrics)
        tf.keras.utils.plot_model(self.model,
                                  "crypto_model.png",
                                  show_shapes=True)

    def trainModel(self, features, labels, validationSplit: float, batchSize: int, epoch: int):
        """Train the model by feeding it data."""


        history = self.model.fit(x=features, y=labels, batch_size=batchSize,
                                 validation_split=validationSplit, epochs=epoch,
                                 shuffle=True)

        # The list of epochs is stored separately from the rest of history.
        epochs = history.epoch

        # To track the progression of training, gather a snapshot
        # of the model's mean squared error at each epoch.
        hist = pd.DataFrame(history.history)
        return epochs, hist

    """
    Evalutaes the model on features.
    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    def evaluate(self, features, label, batchSize: int):
        return self.model.evaluate(features, label, batchSize)

    def simulatePerformanceOnTrades(self, features, labels):
        meanTotalLoss = 0.0
        outputs = self.model.predict(features)
        predictions = []

        for i in range(len(outputs)):
            output = outputs[i][0]
            prediction = self._convertOutputToTradePrediction(output)
            actual = self._convertOutputToTradePrediction(labels[i][0])
            meanTotalLoss += abs(prediction - actual)
            predictions.append(prediction)

        return meanTotalLoss / len(outputs), predictions

    def simulateUpDownPerformance(self, features, labels):
        numCorrect = 0
        outputs = self.model.predict(features)
        for i in range(len(outputs)):
            output = outputs[i][0]
            prediction = self._convertOutputToTradePrediction(output)
            actual = self._convertOutputToTradePrediction(labels[i][0])

            if prediction > 1.0 and actual > 1.0 or prediction <= 1.0 and actual <= 1.0:
                numCorrect += 1

        return numCorrect / len(outputs)

    def makePricePrediction(self, featuresOfDay, currentPrice):
        outputs = self.model.predict(np.asarray([featuresOfDay]))
        prediction = self._convertOutputToTradePrediction(outputs[0][0])
        return currentPrice * prediction

    def plotCurve(self, epochs, hist, metrics):
        """Plot a curve of one or more classification metrics vs. epoch."""
        # list_of_metrics should be one of the names shown in:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()
        plt.show()

    def exportWeights(self):
        self.model.save_weights(self.exportPath)

    def loadWeights(self):
        self.model.load_weights(self.exportPath)

    def _convertOutputToTradePrediction(self, y):
        max = MAX_ALLOWED_CHANGE_FOR_DATASET
        min = MIN_ALLOWED_CHANGE_FOR_DATASET
        diff = max - min
        return y * diff + min

    def _buildMetrics(self):
        self._metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='mean_abs_error')
        ]
        self.listOfMetrics = ["mean_abs_error"]

    def _configureForGPU(self):
        # https://www.tensorflow.org/guide/gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print("Model GPU setup error: " + str(e))
