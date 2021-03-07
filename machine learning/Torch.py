import torch as th

x = th.tensor(1.0)
y = th.tensor(2.0)

w = th.tensor(1.0, requires_grad=True)

y_hat = w * x

loss = (y_hat - y) ** 2
print(loss)

loss.backward()
print(w.grad)