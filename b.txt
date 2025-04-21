import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Create output directory
os.makedirs('models', exist_ok=True)

# 1. Data Preparation
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr  =  LogisticRegression()
lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
print(accuracy_score(y_pred,y_test))
joblib.dump(lr,'models/model1.pkl')


# # Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



lr2 = LogisticRegression(C=2.0)
lr2.fit(X_train_scaled,y_train)
y_pred2 = lr.predict(X_test_scaled)
print(accuracy_score(y_pred2,y_test))

joblib.dump(lr,'models/model2.pkl')