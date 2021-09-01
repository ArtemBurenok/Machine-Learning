from keras.layers import Embedding, Dense, Flatten
from keras.models import Sequential
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

X_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
X_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))

model.add(Flatten())

model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=["acc"])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

