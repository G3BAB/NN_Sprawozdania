from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/Colab Notebooks/waluty.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1').iloc[3:, [1, 2]].dropna()
data.columns = ['Date', 'ExchangeRate']
data['Date'] = pd.to_datetime(data['Date'])
data['ExchangeRate'] = data['ExchangeRate'].astype(float)
data.set_index('Date', inplace=True)

# Przygotowanie danych
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['ExchangeRate']])

train_size = int(len(data_scaled) * 0.80)
train, test = data_scaled[:train_size], data_scaled[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Tworzenie modelu
look_back = 3
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

X_train = X_train.reshape((X_train.shape[0], look_back, 1))
X_test = X_test.reshape((X_test.shape[0], look_back, 1))

model = Sequential([
    LSTM(50, return_sequences=True, activation='linear', input_shape=(look_back, 1)),
    LSTM(50, activation='linear'),
    Dense(1, activation='linear')
])
model.compile(loss='mean_squared_error', optimizer='adam')

# Trenowanie i predykcje
model.fit(X_train, Y_train, epochs=60, batch_size=1, verbose=2)

train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

train_score = np.sqrt(mean_squared_error(Y_train, train_predict))
test_score = np.sqrt(mean_squared_error(Y_test, test_predict))
print(f'Train RMSE: {train_score:.5f}')
print(f'Test RMSE: {test_score:.5f}')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, scaler.inverse_transform(data_scaled), label='Dane')
train_plot = np.empty_like(data_scaled)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back] = train_predict
plt.plot(data.index, train_plot, label='Train Predict')

test_plot = np.empty_like(data_scaled)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (look_back * 2):len(data_scaled)] = test_predict
plt.plot(data.index, test_plot, label='Test Predict')

plt.legend()
plt.show()
