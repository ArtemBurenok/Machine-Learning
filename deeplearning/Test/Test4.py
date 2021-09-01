from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_boston()

target = pd.DataFrame(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)


def cleanData(Frame):
    for col1 in Frame.columns:
        for col2 in Frame.columns:
            if col2 != col1 and np.abs(Frame[col1].corr(Frame[col2])) > 0.8:
                Frame.drop(col1, axis=1)


cleanData(data)
X_train, X_valid, y_train, y_valid = train_test_split(data, target, random_state=32, test_size=0.3)

model = VotingRegressor()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_valid, y_valid))
