from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import keras
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequence(sequence, dimension = 10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(46, activation="softmax"))

model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

model.fit(partial_x_train, partial_y_train, batch_size=512, validation_data=(x_val, y_val), epochs=20)

model.evaluate(partial_x_train, partial_y_train)


