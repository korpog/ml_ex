import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# read data
data = pd.read_excel('data/fruit.xlsx')

le = preprocessing.LabelEncoder()

le.fit(data['Class'])
y = le.transform(data['Class'])

X = data.copy()
X.drop(['Class'], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

model = XGBClassifier(n_estimators=50, learning_rate=0.5, max_depth=2,
                      n_jobs=6, early_stopping_rounds=5, random_state=0)

model.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          verbose=False)

predictions = model.predict(X_valid)

acc = accuracy_score(y_valid, predictions)

print(f"Accuracy: {acc}")