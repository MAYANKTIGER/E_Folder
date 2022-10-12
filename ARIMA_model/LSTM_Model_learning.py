import numpy as np
import datetime
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
#
# # preparing independent and dependent feature
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('E:\\ARIMA_model\\data\\monthlyMilkProduction.csv', index_col='Date', parse_dates=True)
df.index.freq='MS'
#df.columns= ["Date", "Production"]
#df['Date']=pd.to_datetime(df['Date'])
#df.set_index('Date', inplace=True)
#df.sort_index(inplace=True)
#df = df.fillna(df["Production"].mean())

#print(df.head(2))
# df.plot(figsize=(12,6))
# plt.show()
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(df["Production"])
# #result.plot();
# plt.show()
# #print(df.isnull().sum())
# print(df.head(2))
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(df["Production"])
# result.plot();
# plt.show()
#print(df.head(2))
#print(len(df))

# Split data into training part and testing part

train = df.iloc[:156]
test = df.iloc[156:] # last 12 month for testing

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# print(df.head(2))
# print(df.tail(2))

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

#print(scaled_train[:10])
#print(scaled_test)
#print(scaled_train[:4])
# Neural Network Generator
from keras.preprocessing.sequence import TimeseriesGenerator
# n_input = 3
# n_features = 1
# generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)

# X,y = generator[0]
# print(f'Given the Array: \n{X.flatten()}')
# print(f'Predict this y: \n {y}')

#print(X.flatten(),"-->", y.flatten()) convert 2 dim to one

n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())


# fit model
model.fit(generator,epochs=50)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

last_train_batch = scaled_train[-12:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)
print(scaled_test[0])

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred)

    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

print(test_predictions)

# test.head()
#
# true_predictions = scaler.inverse_transform(test_predictions)
#
# test.plot(figsize=(14,5))
#
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# rmse=sqrt(mean_squared_error(test['Production'],test['Predictions']))
# print(rmse)