import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns

train_data = pd.read_csv("C:/Datasets/train.csv")
test_data = pd.read_csv("C:/Datasets/test.csv")
data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

data["Family"] = data.SibSp + data.Parch
data["isAlone"] = data.Family == 0

data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data.Cabin = data.Cabin.fillna('NA')

print(data.nunique())