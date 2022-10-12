"""Building Machine Learning Pipelines: Data Analysis Phase
creating Machine Learning Pipelines

All the Lifecycle In A Data Science Projects
Data Analysis
Feature Engineering
Feature Selection
Model Building
Model Deployment   """



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline -- directly use plt.show()

# display all the columns of the dataset
pd.pandas.set_option('display.max_columns' ,None)

dataset = pd.read_csv('D:\\Learning\\Advance_house_price\\train.csv')
print(dataset.shape) # print shape of dataset with rows and columns

#top 5 records
dataset.head()

"""In Data Analysis We will Analyze To Find out the below stuff
Missing Values
All The Numerical Variables
Distribution of the Numerical Variables
Categorical Variables
Cardinality of Categorical Variables
Outliers
Relationship between independent and dependent feature(SalePrice)
"""
# first we find out the missing values in the dataset

## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values

features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

## 2- step print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')
# will print mean of all the missing values in every columns

# if any dependencies(or ralationship with) of missing values of a row on a dependent variable then we
#will find it first

for feature in features_with_na:
    data = dataset.copy()

 # let's make a variable that indicates 1 if the observation was missing or zero otherwise
data[feature]  = np.where(data[feature].isnull(), 1, 0)

#let's calculate the mean SalePrice where the information is missing or present
data.groupby(feature)['SalePrice'].median().plot.bar()
plt.title(feature)
plt.show()
#Here With the relation between the missing values and the dependent variable is clearly visible.
# So We need to replace these nan values with something meaningful which we will do in the
# Feature Engineering section

# ID is not required then we will delete it
print("Id of Houses {}".format(len(dataset.Id)))

# now will find the numeric values because we have to predict price
# list of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
# here 'o' is object that is strings fields values

#Temporal Variables(Eg: Datetime Variables)
#From the Dataset we have 4 year variables. We have extract information from the datetime

# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature

## let's explore the content of these year variables
for feature in year_feature:
    print(feature, dataset[feature].unique())

## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")

## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    if feature!='YrSold':
        data=dataset.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
# we must have to find discrete variable and continuous variable
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
# it is showing that if length of same data id less than 25 then it is discrete value

discrete_feature
## Lets Find the realtionship between them and Sale PRice
for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
## There is a relationship between variable number and SalePrice
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))

for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()

# here we wil check the normal distribution of continuous variable
# convert non normal dist to normal dist which is best for prediction









