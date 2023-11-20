import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

model = LinearSVC(dual="auto", random_state=0)
data = pd.read_excel('data/fruit.xlsx')

le = preprocessing.LabelEncoder()

le.fit(data['Class'])
y = le.transform(data['Class'])

X = data.copy()
X.drop(['Class'], axis=1, inplace=True)

scores = cross_val_score(model, X, y, cv=5)

for score in scores:
    print(f"Accuracy scores: {scores}")
    print(f"Mean accuracy score: {scores.mean()}")
