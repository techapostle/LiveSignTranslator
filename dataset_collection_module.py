import cv2
import numpy as np
import time
import os
from string import ascii_lowercase
import action_detection_module as ad


class ModelTrainer():
    def __init__(self) -> None:
        self.DATA_PATH = os.path.join('MP_Data')
        arr = []
        for i in range(0, 10):
            arr.append(str(i))
        for c in ascii_lowercase:
            arr.append(c)
        # Actions that we try to detect
        self.actions = np.array(arr)
        # Thirty videos worth of data
        self.no_sequences = 30
        # Videos are going to be 30 frames in length
        self.sequence_length = 30
        # Folder start
        self.start_folder = 1

    def createDirectories(self):
        for action in self.actions:
            for sequence in range(1, self.no_sequences+1):
                try:
                    os.makedirs(os.path.join(
                        self.DATA_PATH, action, str(sequence)))
                except:
                    pass


def main():
    modelTrainer = ModelTrainer()
    modelTrainer.createDirectories()
    cap = cv2.VideoCapture(0)
    holistic = ActionDetector(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    exit = False

    for action in modelTrainer.actions:

        # Loop through sequences aka videos
        start = False

        for sequence in range(modelTrainer.start_folder, modelTrainer.start_folder+modelTrainer.no_sequences):
            if exit:
                break
            # Loop through video length aka sequence length
            for frame_num in range(modelTrainer.sequence_length):
                # Read feed
                ret, frame = cap.read()
                # Make detections
                image, results = holistic.mediapipe_detection(
                    frame, holistic.holistic)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Draw landmarks
                holistic.draw_landmarks_styled(image, results)
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                # NEW Export keypoints
                keypoints = holistic.extract_landmarks(results)
                npy_path = os.path.join(
                    modelTrainer.DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    exit = True
                    break

        if exit:
            break
        while start == False:
            if cv2.waitKey(10) & 0xFF == ord('s'):
                start = True
    cap.release()
    cv2.destroyAllWindows()


main()

