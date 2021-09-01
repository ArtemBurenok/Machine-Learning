from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = load_breast_cancer()

target = pd.DataFrame(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=12)

bag_clf = BaggingClassifier(DecisionTreeClassifier, n_estimators=500, max_samples=100, n_jobs=-1)

bag_clf.fit(X_train, y_train)
print(bag_clf.score(X_train, y_train))
print(bag_clf.score(X_test, y_test))
