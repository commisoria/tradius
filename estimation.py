import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential
from datetime import time,datetime
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

asset='PAXG-USD'

def calc_close(asset):
    end=datetime.now()
    data=yf.download(tickers=asset, start="2016-01-01", end=end)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # learn the scale
    prediction_days = 30
    future_day=0
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create Neural Network

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50))
    model.add(Dropout((0.2)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # test_data

    test_data=yf.download(tickers=asset, start="2020-01-01", end=end)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # prediction

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction1 = scaler.inverse_transform(prediction)
    array=np.array(prediction1)
    estimation=int(round(array[0][0],))
    return estimation

def calc_low(asset):
    end=datetime.now()
    data=yf.download(tickers=asset, start="2016-01-01", end=end)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Low'].values.reshape(-1, 1))  # learn the scale
    prediction_days = 30
    future_day=0
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create Neural Network

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50))
    model.add(Dropout((0.2)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # test_data

    test_data=yf.download(tickers=asset, start="2020-01-01", end=end)
    actual_prices = test_data['Low'].values
    total_dataset = pd.concat((data['Low'], test_data['Low']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # prediction

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction1 = scaler.inverse_transform(prediction)
    array=np.array(prediction1)
    estimation=int(round(array[0][0],))
    return estimation
def calc_high(asset):
    end=datetime.now()
    data=yf.download(tickers=asset, start="2016-01-01", end=end)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['High'].values.reshape(-1, 1))  # learn the scale
    prediction_days = 30
    future_day=0
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create Neural Network

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout((0.2)))
    model.add(LSTM(units=50))
    model.add(Dropout((0.2)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # test_data

    test_data=yf.download(tickers=asset, start="2020-01-01", end=end)
    actual_prices = test_data['High'].values
    total_dataset = pd.concat((data['High'], test_data['High']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # prediction

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction1 = scaler.inverse_transform(prediction)
    array=np.array(prediction1)
    estimation=int(round(array[0][0],))
    return estimation

est=calc_high(asset=asset)
print(est)