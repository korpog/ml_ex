import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


model = LinearRegression()
data = pd.read_csv("data/Admission_Predict.csv", index_col="Serial No.")

features = ['GRE Score', 'TOEFL Score',
            'University Ranking', 'SOP', 'LOR', 'CGPA', 'Research']
label = ['Chance of Admit']

X = data.copy()
y = data[label]
X.drop(label, axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

model.fit(X_train, y_train)

pred = model.predict(X_valid)

print('mean_squared_error : ', mean_squared_error(y_valid, pred))
print('mean_absolute_error : ', mean_absolute_error(y_valid, pred))
