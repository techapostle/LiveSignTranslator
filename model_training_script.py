import numpy as np
import os
import dataset_collection_module as dc
from string import ascii_lowercase
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

data = []
for i in range(0, 10):
    data.append(str(i))
for c in ascii_lowercase:
    data.append(c)

gestureCollector = dc.GestureCollector(data)
label_map = {label: num for num,
             label in enumerate(gestureCollector.actions)}
sequences, labels = [], []

# Preprocess data
for action in gestureCollector.actions:
    for sequence in np.array(os.listdir(
            os.path.join(gestureCollector.DATA_PATH, action))).astype(int):
        window = []
        for frame in range(gestureCollector.sequence_length):
            res = np.load(os.path.join(gestureCollector.DATA_PATH, action, str(
                sequence), "{}.npy".format(frame)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=X.shape))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(gestureCollector.actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
print(model.summary())
