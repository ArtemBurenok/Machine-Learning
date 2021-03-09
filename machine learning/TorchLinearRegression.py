import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

# 0)prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=1, noise=20)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

x = y.view(y.shape[0], 1)

# 1)model
n_sample, n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2)loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3)train
num_epoch = 1000

for epoch in range(num_epoch):
    # prediction and loss
    y_prediction = model(X)
    loss = criterion(y_prediction, y)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

print(loss.item())
