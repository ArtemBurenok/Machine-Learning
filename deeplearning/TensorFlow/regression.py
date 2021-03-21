from keras.datasets import boston_housing
from keras import layers
from keras import models
import numpy as np

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std


def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(test_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss="mse", metrics=["mae"])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []

for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_target[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_target[:i * num_val_samples],
                                           train_target[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
              verbose=0, validation_data=(val_data, val_targets))

    model.evaluate(val_data, val_targets, verbose=0)
