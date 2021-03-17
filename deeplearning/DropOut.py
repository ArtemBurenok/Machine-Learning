from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])

one_hot_label = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_label[i][l] = 1

labels = one_hot_label

test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

relu = lambda x: (x >= 0) * x
relu2 = lambda x: x >= 0
alpha, iteration, hidden_size, pixel_per_image, num_labels = (0.005, 350, 40, 784, 10)

weight0_1 = 0.2 * np.random.random((pixel_per_image, hidden_size)) - 0.1
weight1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iteration):
    error, correct_cnt = (0.0, 0)
    for i in range(len(images)):
        layer0 = images[i:i + 1]
        layer1 = relu(layer0.dot(weight0_1))
        dropout_mask = np.random.randint(2, size=layer1.shape)
        layer1 *= dropout_mask * 2
        layer2 = layer1.dot(weight1_2)

        error += np.sum((labels[i: i + 1] - layer2) ** 2)
        correct_cnt += int(np.argmax(layer2) == np.argmax(labels[i: i + 1]))

        layer_2_delta = labels[i: i + 1] - layer2
        layer_1_delta = layer_2_delta.dot(weight1_2.T) * relu2(layer1)

        layer_1_delta *= dropout_mask

        weight1_2 += alpha * layer1.T.dot(layer_2_delta)
        weight0_1 += alpha * layer0.T.dot(layer_1_delta)

    print("Error: " + str(error / float(len(images))))