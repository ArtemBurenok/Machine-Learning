from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

data = load_boston()
target = pd.DataFrame(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)
data.describe()

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=32, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

y_hat_test = model.predict(X_test)

predictDataFrame = pd.DataFrame(y_hat_test, columns=["Predict"])
