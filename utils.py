import os
import shutil
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

NUMBER_OF_COORDINATES = 84
IMAGES_DATASET_DIR = './dataset'
DATASET_FILE_CSV = './landmarks.csv'
SAVED_MODEL_FILE = './saved_model'
CONVERTED_MODEL_FILE = './tfjs_model'
LANDMARKS_NAMES = [
    'wrist_x_left',
    'wrist_y_left',
    'thumb_cmc_x_left',
    'thumb_cmc_y_left',
    'thumb_mcp_x_left',
    'thumb_mcp_y_left',
    'thumb_ip_x_left',
    'thumb_ip_y_left',
    'thumb_tip_x_left',
    'thumb_tip_y_left',
    'index_finger_mcp_x_left',
    'index_finger_mcp_y_left',
    'index_finger_pip_x_left',
    'index_finger_pip_y_left',
    'index_finger_dip_x_left',
    'index_finger_dip_y_left',
    'index_finger_tip_x_left',
    'index_finger_tip_y_left',
    'middle_finger_mcp_x_left',
    'middle_finger_mcp_y_left',
    'middle_finger_pip_x_left',
    'middle_finger_pip_y_left',
    'middle_finger_dip_x_left',
    'middle_finger_dip_y_left',
    'middle_finger_tip_x_left',
    'middle_finger_tip_y_left',
    'ring_finger_mcp_x_left',
    'ring_finger_mcp_y_left',
    'ring_finger_pip_x_left',
    'ring_finger_pip_y_left',
    'ring_finger_dip_x_left',
    'ring_finger_dip_y_left',
    'ring_finger_tip_x_left',
    'ring_finger_tip_y_left',
    'pinky_finger_mcp_x_left',
    'pinky_finger_mcp_y_left',
    'pinky_finger_pip_x_left',
    'pinky_finger_pip_y_left',
    'pinky_finger_dip_x_left',
    'pinky_finger_dip_y_left',
    'pinky_finger_tip_x_left',
    'pinky_finger_tip_y_left',
    'wrist_x_right',
    'wrist_y_right',
    'thumb_cmc_x_right',
    'thumb_cmc_y_right',
    'thumb_mcp_x_right',
    'thumb_mcp_y_right',
    'thumb_ip_x_right',
    'thumb_ip_y_right',
    'thumb_tip_x_right',
    'thumb_tip_y_right',
    'index_finger_mcp_x_right',
    'index_finger_mcp_y_right',
    'index_finger_pip_x_right',
    'index_finger_pip_y_right',
    'index_finger_dip_x_right',
    'index_finger_dip_y_right',
    'index_finger_tip_x_right',
    'index_finger_tip_y_right',
    'middle_finger_mcp_x_right',
    'middle_finger_mcp_y_right',
    'middle_finger_pip_x_right',
    'middle_finger_pip_y_right',
    'middle_finger_dip_x_right',
    'middle_finger_dip_y_right',
    'middle_finger_tip_x_right',
    'middle_finger_tip_y_right',
    'ring_finger_mcp_x_right',
    'ring_finger_mcp_y_right',
    'ring_finger_pip_x_right',
    'ring_finger_pip_y_right',
    'ring_finger_dip_x_right',
    'ring_finger_dip_y_right',
    'ring_finger_tip_x_right',
    'ring_finger_tip_y_right',
    'pinky_finger_mcp_x_right',
    'pinky_finger_mcp_y_right',
    'pinky_finger_pip_x_right',
    'pinky_finger_pip_y_right',
    'pinky_finger_dip_x_right',
    'pinky_finger_dip_y_right',
    'pinky_finger_tip_x_right',
    'pinky_finger_tip_y_right',
]


def create_dir_if_not_exists(dirname):
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


def convert_to_txt(landmarks):
  with open('landmarks.txt', "w") as file:
    for landmark in landmarks:
      file.write(str(landmark))
      file.write("\n")


def landmarks_to_csv(landmarks):
  convert_to_txt(landmarks)
  data = np.array(landmarks)
  columns = [
      'signal',
  ]

  columns.extend(LANDMARKS_NAMES)

  df = pd.DataFrame(data, columns=columns)
  df.to_csv(DATASET_FILE_CSV, index=False)


def landmark_coordinates(landmarks):
  coordinates = []
  # x_coordinates = []
  # y_coordinates = []

  for hand_landmarks in landmarks:
    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      coordinates.append(x)
      coordinates.append(y)

    # for i in range(len(hand_landmarks.landmark)):
    #   x = hand_landmarks.landmark[i].x
    #   y = hand_landmarks.landmark[i].y

    #   coordinates.append(x - min(x_coordinates))
    #   coordinates.append(y - min(y_coordinates))

  return coordinates


def coordinates_to_input_data(coordinates):
  input_data = {}

  for i in range(len(coordinates)):
    input_data[LANDMARKS_NAMES[i]] = np.array([coordinates[i]])

  return input_data


def utils_save_image(dirname, filename, frame):
  create_dir_if_not_exists(os.path.join(IMAGES_DATASET_DIR, dirname))

  cv2.imwrite(os.path.join(IMAGES_DATASET_DIR, dirname, filename), frame)
