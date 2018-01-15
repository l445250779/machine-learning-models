# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:42:03 2018

@author: lixud
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3]) # transform the states names to numbers
#Dummy Encoding the States
onehotencoder = OneHotEncoder(categorical_features = [3]) #encode the first column
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# Add the constant for the b0 (add a column of 1)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #x2 has the highest P-value, eliminated x2

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #eliminated 1

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #eliminated 4

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #eliminated 5

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #finished

#The strongest variable for profit impact is the R&D spent

#only use the data after backward elimination
X_train_OLS, X_test_OLS, y_train_OLS, y_test_OLS = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
y_pred_OLS = regressor_OLS.predict(X_test_OLS)

#compare the performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mse_OLS = mean_squared_error(y_test, y_pred_OLS)
