import cv2
import os
import mediapipe as mp
import utils
from signals import SIGNALS

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

DATASET_DIR = './dataset'
dataset_landmarks = []

for signal_dir in os.listdir(DATASET_DIR):
  for image_file_path in os.listdir(os.path.join(DATASET_DIR, signal_dir)):
    print(
        f'Collecting data from the signal of {signal_dir} and image of index {image_file_path}')

    image = cv2.imread(os.path.join(DATASET_DIR, signal_dir, image_file_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
      coordinates = utils.landmark_coordinates(results.multi_hand_landmarks)

      if len(coordinates) == utils.NUMBER_OF_COORDINATES:
        coordinates.insert(0, SIGNALS.index(signal_dir))
        dataset_landmarks.append(coordinates)

utils.landmarks_to_csv(dataset_landmarks)
print('Done!')
