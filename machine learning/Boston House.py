import torch as th
import torch.nn as nn

x = th.tensor([[1], [2], [3], [4]], dtype=th.float32)
y = th.tensor([[2], [4], [6], [8]], dtype=th.float32)

n_sample, n_features = x.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

loss = nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(), lr=0.01)

for i in range(100):
    y_prediction = model(x)

    l = loss(y, y_prediction)

    l.backward()

    optimizer.step()
    optimizer.zero_grad()
    print(l)