import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import os
import time

class ActionDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                   enable_segmentation=False, smooth_segmentation=True, refine_face_landmarks=False,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation,
            self.smooth_segmentation, self.refine_face_landmarks, self.min_detection_confidence,
            self.min_tracking_confidence
        )

    def mediapipe_detection(self, frame, model):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        results = model.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS) # Draw face connections
        # Can also use FACEMESH_TESSELATION instead as shown below
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    def draw_landmarks_styled(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                 self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

    def extract_landmarks(self, results):
        # Extract face, pose, left hand, and right hand landmarks into separate arrays called face, pose, left_hand, and right_hand.
        # If there are no landmarks detected, then the arrays will be populated with 0s to the length of their respective number of landmarks.
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        left_hand = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        right_hand = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        # Return concatenated numpy array containing all extracted landmarks.
        return np.concatenate([pose, face, left_hand, right_hand])


def main():
    # # Path for exported data
    # DATA_PATH = os.path.join('MP_DATA')
    # # Define actions to train and detect
    # actions = np.array(['hello', 'thanks', 'iloveyou'])
    # # Number of sequences to capture for each action
    # no_sequences = 30
    # # Number of frames per sequence
    # sequence_length = 30

    cap = cv2.VideoCapture(0)

    # Initialize ActionDetector object
    holistic = ActionDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # for FPS counter
    prev_time = 0
    curr_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture image...aborting.')
            break

        image, results = holistic.mediapipe_detection(frame, holistic.holistic)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw landmarks
        holistic.draw_landmarks_styled(image, results)
        # print(results.pose_landmarks)

        # Extract landmarks
        # landmarks = holistic.extract_landmarks(results).shape()

        # record FPS
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        # putText takes image, text, pos, font, scale, color
        cv2.putText(image, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))

        cv2.imshow("Camera Feed", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
