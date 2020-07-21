# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from numpy import savetxt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# load dataset
'''dataset = pd.read_csv('./data/q2_dataset.csv')

# select open, high, low, and volume as features
X_orig = dataset.iloc[:, 2:6].values
# select open as labels
Y_orig = dataset.iloc[:, 3:4].values

# split data into training and testing using 0.7 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, train_size = 0.7, test_size=0.3, shuffle=False)

# save training set
train_data = np.column_stack((X_train, Y_train))
np.savetxt("./data/train_data_RNN.csv", train_data, delimiter=",")
# save testing set
test_data = np.column_stack((X_test, Y_test))
np.savetxt("./data/test_data_RNN.csv", test_data, delimiter=",")'''


if __name__ == "__main__":
    # 1. load your training data

    # from ./data load train_data_RNN.csv
    dataset = pd.read_csv('./data/train_data_RNN.csv')

    # select features
    X_train_orig = dataset.iloc[:, 0:4].values
    # select labels
    Y_train_orig = dataset.iloc[:, 4:5].values

    # normalize data using minmaxscaler
    min_max_scaler = MinMaxScaler()
    X_orig = min_max_scaler.fit_transform(X_train_orig)

    # covert data into time series and supervised
    # create empty list to store feature and label
    features = []
    labels = []
    # choose 3 as window size (using 3 days feature)
    window_size = 3

    # generate data
    for i in range(4, len(X_orig) + 1):
        # using the prev three days open/high/low/volume as features
        features.append(X_orig[i - window_size:i, :])
        # using the next day's open price as label
        labels.append(Y_train_orig[i - window_size - 1, 0])

    # change to numpy array
    X_train = np.array(features)
    Y_train = np.array(labels)
    # 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

    # create model
    model = Sequential()
    # using LSTM as the first layer
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    # add a dropout layer to avoid overfitting
    #model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu'))
    # output layer
    model.add(Dense(1, activation='linear'))
    # compile model
    model.compile(optimizer="rmsprop", loss="mae")

    history = model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2, shuffle=True)

    # 3. Save your model
    model.save('20862738_RNN_model.h5')
