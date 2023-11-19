import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=6)
data = pd.read_excel("data/fruit.xlsx")

X = data.copy()
y = data['Class']
X.drop('Class', axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

model.fit(X_train, y_train)
pred = model.predict(X_valid)


print(accuracy_score(y_valid, pred))
