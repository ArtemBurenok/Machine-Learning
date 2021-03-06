import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(200):
    y_prediction = model(X)
    loss = criterion(y_prediction, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(loss.item())




