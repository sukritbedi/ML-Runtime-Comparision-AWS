import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timeit

start = timeit.default_timer()
dataset = pd.read_csv('dataset_lin.csv')
dataset.drop(dataset.columns[[0,1,5,6]], axis=1, inplace=True)

x = dataset.iloc[:, :3].values
y = dataset.iloc[:, 3].values
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 100)

regressionModel = LinearRegression()
regressionModel.fit(X_train,y_train)

y_pred = regressionModel.predict(X_test)

error = 0
for i in range(len(y_pred)):
       int_error = abs((y_pred[i] - y_test[i])/y_test[i])
       error += int_error

error = error/len(y_pred)

print("The % error of the given model is : " + str(error*100) + "%")

stop = timeit.default_timer()

print('Time: ', stop - start)
