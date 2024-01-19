import time
import utils
from signals import SIGNALS

dataset_size = 100
dataset_landmarks = []


def collect_signals_data():
  for signal in SIGNALS:
    print(f'Collecting the data from the signal of {signal}')
    title = f'Press "Q" to start and make the signal of "{signal}"'

    while True:
      frame = utils.image_frame()
      landmarks = utils.hands_landmarks(frame)

      if landmarks:
        utils.draw_landmarks(frame, landmarks)

      utils.draw_title(frame, title)
      utils.show_image_frame(frame)

      if utils.is_letter_q_pressed():
        collect_signal_landmarks(signal)
        break


def collect_signal_landmarks(signal):
  counter = 0
  time.sleep(3)

  while counter < dataset_size:
    frame = utils.image_frame()
    landmarks = utils.hands_landmarks(frame)

    if landmarks:
      coordinates = utils.landmark_coordinates(landmarks)

      if len(coordinates) == 84:
        coordinates.insert(0, signal)
        dataset_landmarks.append(coordinates)
        counter += 1

    utils.show_image_frame(frame)
    utils.wait_letter_q_pressed()


collect_signals_data()

utils.destroy_capture_windows()
utils.landmarks_to_csv(dataset_landmarks)

print('Done!')
