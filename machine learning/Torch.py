import torch as th
import torch.nn as nn

x = th.tensor([[1], [2], [3], [4]], dtype=th.float32)
y = th.tensor([[2], [4], [6], [8]], dtype=th.float32)

x_test = th.tensor([5], dtype=th.float32)

n_sample, n_features = x.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size) # Построенние модели

print(model(x_test).item())

loss = nn.MSELoss() # функция потерь
optimizer = th.optim.SGD(model.parameters(), lr=0.01) # оптимизация

for i in range(100):
    y_pred = model(x)

    l = loss(y, y_pred)

    l.backward() # Вычесление градиента

    optimizer.step() # Изменяем веса

    optimizer.zero_grad()
print(model(x_test).item())