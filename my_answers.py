# import keras
import numpy as np

# from keras.layers import LSTM, Dense
# from keras.models import Sequential


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    """Containers for input/output pairs based on window size.
    >>> X, y = window_transform_series(np.array([1, 3, 5, 7, 9, 11, 13]), 2)
    >>> X
    array([[ 1,  3],
           [ 3,  5],
           [ 5,  7],
           [ 7,  9],
           [ 9, 11]])
    >>> y
    array([[ 5],
           [ 7],
           [ 9],
           [11],
           [13]])
    """
    n = len(series)
    y = [[x] for x in series[window_size:]]
    X = [series[i:i + window_size] for i in range(0, n - window_size)]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    pass


# TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # punctuation = ['!', ',', '.', ':', ';', '?']

    return text


# TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs, outputs


# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    pass
