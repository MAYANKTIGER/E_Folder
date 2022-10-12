import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\certainty\\ARIMA\\perrin-freres-monthly-champagne.csv')
df.columns = ["Month", "Sales"]
df.drop(106, axis=0, inplace=True)
df.drop(105,axis=0,inplace=True)
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Sales'])
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['Sales'])

df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
adfuller_test(df['Seasonal First Difference'].dropna())
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()
print(model_fit.summary())

