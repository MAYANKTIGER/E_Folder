import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import data/read csv file
data = pd.read_csv('D:\\ML_projects\\Human Life Span Prediction.csv')
data.head()


# task to be perform :
# EDA
# Featrure Engineering
# Feature Selection
# Model Fit
# Prediction

# EDA : ->>
# to get the full info of the dataset/
#data.info()
""" it is clearly showing that some missing values are there """

# to check missing values
#data.isnull().sum()
# features holds some missing values that will affect our model that we have handle it

# Age(Life expectancy) is our dependent variable that we have to predict
# univariate visualization

#sns.jointplot(x=data['Alcohol'], y= data['Age'],data=data)
# analyze the relationship between two variables
# dependency of a independent variable on dependent variable
# diagram shows no a big impact of alcohol on Age


sns.jointplot(x=data['GDP'], y= data['Age'],data=data,kind='reg')





