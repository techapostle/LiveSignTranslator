import numpy as np
import os
import dataset_collection_module as gc
from string import ascii_lowercase
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


class ModelTraining():
    def __init__(self, data) -> None:
        self.gestureCollector = gc.GestureCollector(data)
        self.label_map = {label: num for num,
                          label in enumerate(self.gestureCollector.actions)}
        self.sequences, self.labels = [], []
        self.model = Sequential()

    def preprocessData(self):
        for action in self.gestureCollector.actions:
            for sequence in np.array(os.listdir(os.path.join(self.gestureCollector.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(self.gestureCollector.sequence_length):
                    res = np.load(os.path.join(self.gestureCollector.DATA_PATH, action, str(
                        sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                self.sequences.append(window)
                self.labels.append(self.label_map[action])

    def createLabelsAndFeatures(self):
        self.X = np.array(self.sequences, dtype=type(self.sequences))
        self.y = to_categorical(self.labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.05)
        # self.X_train = np.int64(self.X_train)
        # self.X_test = np.int64(self.X_test)
        # self.y_train = np.int64(self.y_train)
        # self.y_test = np.int64(self.y_test)

    def setupTensorBoard(self):
        log_dir = os.path.join('Logs')
        self.tb_callback = TensorBoard(log_dir=log_dir)

    def setupLSTM(self):
        self.model.add(LSTM(64, return_sequences=True,
                       activation='relu', input_shape=self.X.shape))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(
            Dense(self.gestureCollector.actions.shape[0], activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[
                           'categorical_accuracy'])

    def trainModel(self):
        self.model.fit(self.X_train, self.y_train, epochs=2000,
                       callbacks=[self.tb_callback])

    def saveModel(self):
        self.model.save('action.h5')


def main():
    # Create data array
    data = []
    for i in range(0, 10):
        data.append(str(i))
    for c in ascii_lowercase:
        data.append(c)

    # Initialize model, preprocess, and train
    modelTraining = ModelTraining(data)
    modelTraining.preprocessData()
    modelTraining.createLabelsAndFeatures()
    modelTraining.setupTensorBoard()
    modelTraining.setupLSTM()
    modelTraining.trainModel()
    modelTraining.saveModel()


main()
