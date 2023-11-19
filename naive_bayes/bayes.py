import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = GaussianNB()
data = pd.read_excel('data/fruit.xlsx')

le = preprocessing.LabelEncoder()

le.fit(data['Class'])
y = le.transform(data['Class'])

X = data.copy()
X.drop(['Class'], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

acc = accuracy_score(y_valid, predictions)

print(f"Accuracy: {acc}")
