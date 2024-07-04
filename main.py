import csv
import copy
import itertools
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(self, model_path="model/keypoint.tflite", num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index

def normalize(n, m):
    return n/m

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)

    base_x = 0
    base_y = 0
    for index, landmark_point in enumerate(temp):
        if index == 0:
            base_x = landmark_point[0]
            base_y = landmark_point[1]

        temp[index][0] = temp[index][0] - base_x
        temp[index][1] = temp[index][1] - base_y

    temp = list(itertools.chain.from_iterable(temp))

    max_value = max(list(map(abs, temp)))

    temp = list(map(lambda x: x/max_value, temp))

    return temp

    


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
            )

    keypoint_classifier = KeyPointClassifier()

    with open("model/keypoint_label.csv", encoding="utf-8-sig") as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    image = cv.imread("hand.jpg")
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            print(handedness)

    else:
        print("Nothing to do")

if __name__ == "__main__":
    main()
