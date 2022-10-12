# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller
# import matplotlib.pyplot as plt
#
# #data Import
# df = pd.read_excel("E:\\ARIMA_model\\data\\Series_2.xlsx")
# print(df.head())
#
# # plt.plot(df["Value"])
# # plt.show()
#
# # x= df['Value'].values # to make it in series convert it into one list(df.values)
# # result = adfuller(x)
# # print(result)
# # #print()
# #print()
# X = df["Value"].values
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t%s: %.3f' % (key, value))
#
# if result[0] < result[4]["5%"]:
#     print ("Reject Ho - Time Series is Stationary")
# else:
#     print ("Failed to Reject Ho - Time Series is Non-Stationary")
#
#
# df['diff_values'] = df['Value'].diff()
#
#
# import statsmodels as sm
# print(sm.__version__)
import nltk
x = nltk.word_tokenize("I have a car")
print(x)