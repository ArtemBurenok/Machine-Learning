import numpy as np
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = load_breast_cancer()
target = pd.DataFrame(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=12)

X_train = StandardScaler().fit_transform(X_train)
modelTree = DecisionTreeClassifier()
modelTree.fit(X_train, y_train)

score = cross_val_score(modelTree, X_test, y_test, cv=5)
print(np.mean(score))

modelLinear = LogisticRegression()
modelLinear.fit(X_train, y_train)
scoreLinear = cross_val_score(modelLinear, X_test, y_test, cv=5)
print(np.mean(scoreLinear))

modelSVM = NuSVC()
modelSVM.fit(X_train, y_train)
scoreSVM = cross_val_score(modelSVM, X_test, y_test, cv=5)
print(np.mean(scoreSVM))
