import math
import re

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    w = window_size

    # containers for input/output pairs
    y = [[x] for x in series[w:]]
    X = [series[i:i + w] for i in range(0, len(series) - w)]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    # an LSTM module with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # a Dense module with one unit
    model.add(Dense(1))

    return model


# TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    return re.sub(r"[^a-z!,.:;?]", " ", text)


# TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    w = window_size

    # containers for input/output pairs
    inputs = [text[i:i + w] for i in range(0, len(text) - w, step_size)]
    outputs = [text[i + w] for i in range(0, len(text) - w, step_size)]

    return inputs, outputs


# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    # an LSTM module with 200 hidden units
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # a Dense module with num chars units
    model.add(Dense(num_chars, activation='softmax'))

    return model
