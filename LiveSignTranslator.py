import time
import cv2
import numpy as np
import os
import dataset_collection_module as gc
import action_detection_module as ad
from string import ascii_lowercase
from sklearn.model_selection import train_test_split

# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard
# WORKAROUND
from keras.api._v2.keras.utils import to_categorical
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import LSTM, Dense
from keras.api._v2.keras.callbacks import TensorBoard
from keras.api._v2.keras.models import load_model


class ModelTraining:
    def __init__(self, data) -> None:
        self.gestureCollector = gc.GestureCollector(data)
        self.label_map = {
            label: num for num, label in enumerate(self.gestureCollector.actions)
        }
        self.sequences, self.labels = [], []
        self.model = Sequential()

    def preprocessData(self):
        for action in self.gestureCollector.actions:
            for sequence in np.array(
                os.listdir(os.path.join(self.gestureCollector.DATA_PATH, action))
            ).astype(int):
                window = []
                for frame_num in range(self.gestureCollector.sequence_length):
                    res = np.load(
                        os.path.join(
                            self.gestureCollector.DATA_PATH,
                            action,
                            str(sequence),
                            "{}.npy".format(frame_num),
                        )
                    )
                    window.append(res)
                self.sequences.append(window)
                self.labels.append(self.label_map[action])

    def createLabelsAndFeatures(self):
        self.X = np.array(self.sequences, dtype=type(self.sequences))
        self.y = to_categorical(self.labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.05
        )
        # self.X_train = np.int64(self.X_train)
        # self.X_test = np.int64(self.X_test)
        # self.y_train = np.int64(self.y_train)
        # self.y_test = np.int64(self.y_test)

    def setupTensorBoard(self):
        log_dir = os.path.join("Logs")
        self.tb_callback = TensorBoard(log_dir=log_dir)

    def setupLSTM(self):
        self.model.add(
            LSTM(64, return_sequences=True, activation="relu", input_shape=self.X.shape)
        )
        self.model.add(LSTM(128, return_sequences=True, activation="relu"))
        self.model.add(LSTM(64, return_sequences=False, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(
            Dense(self.gestureCollector.actions.shape[0], activation="softmax")
        )
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

    def trainModel(self):
        self.model.fit(
            self.X_train, self.y_train, epochs=2000, callbacks=[self.tb_callback]
        )

    def saveModel(self):
        self.model.save("action.h5")

    def loadModel(self):
        if self.model:
            del self.model
        self.model = load_model("action.h5")


def main():
    # Create data array
    data = []
    for i in range(0, 10):
        data.append(str(i))
    for c in ascii_lowercase:
        data.append(c)

    # Initialize model, preprocess, and train
    modelTraining = ModelTraining(data)

    # Ask if the user wants to train the model or load a pre-trained model
    print("Would you like to train the model or load a pre-trained model?")
    print("1. Train the model")
    print("2. Load a pre-trained model")
    choice = input("Enter your choice: ")
    if choice == "1":
        modelTraining.preprocessData()
        modelTraining.createLabelsAndFeatures()
        modelTraining.setupTensorBoard()
        modelTraining.setupLSTM()
        modelTraining.trainModel()
        modelTraining.saveModel()
    elif choice == "2":
        modelTraining.loadModel()
        # ask if the user wants to run the model on a webcam feed
        print("Would you like to run the model on a webcam feed?")
        print("1. Yes")
        print("2. No")
        choice = input("Enter your choice: ")
        if choice == "1":
            # run the model on a webcam feed
            print("Running the model on a webcam feed...")
            cap = cv2.VideoCapture(0)
            # Initialize ActionDetector object
            holistic = ad.ActionDetector(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            # for FPS counter
            prev_time = 0
            curr_time = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image...aborting.")
                    break

                image, results = holistic.mediapipe_detection(frame, holistic.holistic)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Draw landmarks
                holistic.draw_landmarks_styled(image, results)
                # print(results.pose_landmarks)

                # Extract landmarks
                # landmarks = holistic.extract_landmarks(results).shape()
                print(holistic.extract_landmarks(results))

                # record FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                # putText takes image, text, pos, font, scale, color
                cv2.putText(
                    image,
                    str(int(fps)),
                    (10, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 0, 0),
                    3,
                )

                cv2.imshow("Action Detection Feed", image)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows
        elif choice == "2":
            # exit the program
            print("Exiting program.")
            exit()
    else:
        print("Invalid choice. Exiting program.")
        exit()


main()
