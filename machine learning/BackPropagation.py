import numpy as np

np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2(x):
    return x > 0


input = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1]])

output = np.array([1, 1, 0, 0]).T

weight0_1 = 2 * np.random.random((3, 4)) - 1
weight1_2 = 2 * np.random.random((4, 1)) - 1
alpha = 0.1

for iteration in range(200):
    Layer2Error = 0
    for i in range(len(input)):
        layer0 = input[i:i + 1]
        layer1 = relu(np.dot(layer0, weight0_1))
        layer2 = np.dot(layer1, weight1_2)

        Layer2Error += np.sum((layer2 - output[i:i + 1]) ** 2)

        Layer2Delta = output[i:i + 1] - layer2
        Layer1Delta = Layer2Delta.dot(weight1_2.T) * relu2(layer1)

        weight0_1 += alpha * layer0.T.dot(Layer1Delta)
        weight1_2 += alpha * layer1.T.dot(Layer2Delta)

    if iteration % 10 == 9:
         print("Error" + str(Layer2Error))
