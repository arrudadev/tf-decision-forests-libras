import os
import shutil
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

DATASET_FILE_CSV = './dataset_landmarks.csv'
SAVED_MODEL_FILE = './saved_model'
CONVERTED_MODEL_FILE = './tfjs_model'


def create_dir(dirname, replace=False):
  if replace and os.path.exists(dirname):
    shutil.rmtree(dirname)

  if not os.path.exists(dirname):
    os.makedirs(dirname)


def hands_landmarks(frame):
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)

  return results.multi_hand_landmarks


def draw_landmarks(frame, landmarks):
  for landmark in landmarks:
    mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)


def draw_title(frame, title):
  cv2.putText(frame, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
              1.0, (0, 0, 0), 3, cv2.LINE_AA)


def image_frame():
  _, frame = capture.read()

  return frame


def show_image_frame(frame):
  cv2.imshow('frame', frame)


def wait_letter_q_pressed():
  letter_q_key = 25

  return cv2.waitKey(letter_q_key)


def is_letter_q_pressed():
  letter_q_key = 25

  return cv2.waitKey(letter_q_key) == ord('q')


def destroy_capture_windows():
  capture.release()
  cv2.destroyAllWindows()


def create_pickle(filename, data):
  f = open(filename, 'wb')
  pickle.dump(data, f)
  f.close()


def landmarks_to_csv(landmarks):
  data = np.array(landmarks)
  columns = [
      'signal',
      'wrist_x',
      'wrist_y',
      'thumb_cmc_x',
      'thumb_cmc_y',
      'thumb_mcp_x',
      'thumb_mcp_y',
      'thumb_ip_x',
      'thumb_ip_y',
      'thumb_tip_x',
      'thumb_tip_y',
      'index_finger_mcp_x',
      'index_finger_mcp_y',
      'index_finger_pip_x',
      'index_finger_pip_y',
      'index_finger_dip_x',
      'index_finger_dip_y',
      'index_finger_tip_x',
      'index_finger_tip_y',
      'middle_finger_mcp_x',
      'middle_finger_mcp_y',
      'middle_finger_pip_x',
      'middle_finger_pip_y',
      'middle_finger_dip_x',
      'middle_finger_dip_y',
      'middle_finger_tip_x',
      'middle_finger_tip_y',
      'ring_finger_mcp_x',
      'ring_finger_mcp_y',
      'ring_finger_pip_x',
      'ring_finger_pip_y',
      'ring_finger_dip_x',
      'ring_finger_dip_y',
      'ring_finger_tip_x',
      'ring_finger_tip_y',
      'pinky_finger_mcp_x',
      'pinky_finger_mcp_y',
      'pinky_finger_pip_x',
      'pinky_finger_pip_y',
      'pinky_finger_dip_x',
      'pinky_finger_dip_y',
      'pinky_finger_tip_x',
      'pinky_finger_tip_y'
  ]

  df = pd.DataFrame(data, columns=columns)
  df.to_csv(DATASET_FILE_CSV, index=False)


def landmark_coordinates(landmarks):
  coordinates = []
  x_coordinates = []
  y_coordinates = []

  for hand_landmarks in landmarks:
    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      x_coordinates.append(x)
      y_coordinates.append(y)

    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      coordinates.append(x - min(x_coordinates))
      coordinates.append(y - min(y_coordinates))

  return coordinates
