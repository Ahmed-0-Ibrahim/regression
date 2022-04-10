import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import linear_model

print("Assignment 2 ML")

#Read Data
data = pd.read_csv('assignment2_dataset_cars.csv')

#view Data
print(data.head())

#check Data
print(data.isnull().sum())

#read data
x = data.iloc[:, :-1]   #Futures
y = data.iloc[:, -1]    #Goal

x_obj = x.select_dtypes(include=["object"])
x_non_obj = x.select_dtypes(exclude=["object"])
print('------------------------------------------------')
print(x_obj)
print(x_non_obj)
print('------------------------------------------------')
la = LabelEncoder()

for i in range (x_obj.shape[1]):
    x_obj.iloc[:,i] = la.fit_transform(x_obj.iloc[:,i])
    
X = pd.concat([x_non_obj,x_obj],axis=1)
#futures Normilzation
print(X)
X=(X-X.mean())/(X.max()-X.min())
print(X)

# #Get the correlation between the features
# corr = data.corr()
# #Top 50% Correlation training features with the Value
# top_feature = corr.index[abs(corr['price'])>0.1]
# #Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = data[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# top_feature = top_feature.delete(-1)
# X = X[top_feature]
# print('************************************')
# print(X)

x_train ,  x_test , y_train , y_test = train_test_split(X,y,train_size =0.8,shuffle=True)

# MV_LR = LinearRegression().fit(x_train, y_train)
# print('Mean Square Error for Training', metrics.mean_squared_error(y_train, MV_LR.predict(x_train)))
# print('Mean Square Error for Testing', metrics.mean_squared_error(y_test, MV_LR.predict(x_test)))

# print(MV_LR.score(x_train, y_train))
# print(MV_LR.score(x_test, y_test))

poly_features = PolynomialFeatures(degree=5)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(x_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
# ypred=poly_model.predict(poly_features.transform(x_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(x_test))


# print('Co-efficient of linear regression',poly_model.coef_)

print('Mean Square Error for test', metrics.mean_squared_error(y_test, prediction))
print('Mean Square Error for training', metrics.mean_squared_error(y_train, y_train_predicted))


