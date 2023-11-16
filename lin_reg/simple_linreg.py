import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = LinearRegression()
data = pd.read_csv("data/weatherHistory.csv")

X = data['Temperature (C)']
X = np.array(X).reshape(-1, 1)
y = data['Humidity']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

model.fit(X_train, y_train)

pred = model.predict(X_valid)

mse =  mean_squared_error(y_valid, pred)
mae = mean_absolute_error(y_valid, pred)
r2 = r2_score(y_valid, pred)

print(f'mean_squared_error : {mse}')
print(f'mean_absolute_error : {mae}')
print(f'R2 score : {r2}')
print (model.coef_)
print (model.intercept_)