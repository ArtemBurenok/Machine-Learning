from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras_preprocessing import sequence
from keras.datasets import imdb

import matplotlib.pyplot as plt

max_features = 10000
max_len = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features) # Выгружаем данные
input_train = sequence.pad_sequences(input_train, maxlen=max_len) # Форматируем
input_test = sequence.pad_sequences(input_test, maxlen=max_len) # Форматируем
# Строим модель

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(batch_size))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2) # Тренируем

# Выводим грфики потерь и точности
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Loss function")
plt.legend()

plt.show()
