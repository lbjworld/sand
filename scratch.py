import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
# For a single-input model with 10 classes (categorical classification):

batch_size=20
timesteps=20
data_dim=6

model = keras.models.Sequential()
model.add(
    keras.layers.recurrent.LSTM(
        64, input_shape=(timesteps, data_dim), return_sequences=True
    )
)
model.add(
    keras.layers.recurrent.LSTM(
        32, return_sequences=True
    )
)
model.add(
    keras.layers.recurrent.LSTM(16)
)
model.add(keras.layers.core.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((2000, timesteps, data_dim))
labels = np.random.randint(3, size=(2000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=3)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=100, batch_size=batch_size)
