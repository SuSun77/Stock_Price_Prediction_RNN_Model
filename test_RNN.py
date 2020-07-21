# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    # 1. Load your saved model
    model = load_model('./models/20862738_RNN_model.h5')


    # 2. Load your testing data
    # load testing data
    dataset = pd.read_csv('./data/test_data_RNN.csv')
    X_test_orig = dataset.iloc[:, 0:4].values
    Y_test_orig = dataset.iloc[:, 4:5].values

    # load training data to obtain same scaler
    dataset = pd.read_csv('./data/train_data_RNN.csv')
    X_train_orig = dataset.iloc[:, 0:4].values

    # normalize testing data
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X_train_orig)
    X_orig = min_max_scaler.transform(X_test_orig)

    features = []
    labels = []
    window_size = 3
    for i in range(4, len(X_orig) + 1):
        features.append(X_orig[i - window_size:i, :])
        labels.append(Y_test_orig[i - window_size - 1, 0])

    # change to numpy array
    X_test = np.array(features)
    Y_test = np.array(labels)


    # 3. Run prediction on the test data and output required plot and loss
    prediction = model.predict(X_test)
    plt.plot(Y_test, label="True")
    plt.plot(prediction, label="Predicted")
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    MAE = mean_absolute_error(Y_test, prediction)
    print('The loss is: {}'.format(MAE))