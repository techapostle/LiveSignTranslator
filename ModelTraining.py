import action_detection_module as ad
import dataset_collection_module as dc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard



class ModelTraining():
    def __init__(self) -> None:
        self.modelTrainer = dc.DataCollector()
        self.label_map = {label:num for num, label in enumerate(self.modelTrainer.actions)}
        self.sequences, self.labels = [], []


    def preprocessData(self):
        for action in self.modelTrainer.actions:
            for sequence in ad.np.array(ad.os.listdir(ad.os.path.join(self.modelTrainer.DATA_PATH, action))).astype(int):
                window = []
                for frame_num in range(self.modelTrainer.sequence_length):
                    res = ad.np.load(ad.os.path.join(self.modelTrainer.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                self.sequences.append(window)
                self.labels.append(self.label_map[action])
    
    def createLabelsAndFeatures(self):
        X = ad.np.array(self.sequences)
        y = to_categorical(self.labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)

    def setupTensorBoard(self):
        log_dir = ad.os.path.join('Logs')
        self.tb_callback = TensorBoard(log_dir=log_dir)

    def setupLSTM(self):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.modelTrainer.actions.shape[0], activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    
    def trainModel(self):
        self.model.fit(self.X_train, self.y_train, epochs=2000, callbacks=[self.tb_callback])

    def saveModel(self):
        self.model.save('action.h5')


def main():
    modelTraining = ModelTraining()

    modelTraining.preprocessData()
    modelTraining.createLabelsAndFeatures()
    modelTraining.setupTensorBoard()
    modelTraining.setupLSTM()
    modelTraining.setupLSTM()
    modelTraining.trainModel()
    modelTraining.saveModel()



main()