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

      utils.draw_landmarks(frame)
      utils.draw_title(frame, title)
      utils.show_image_frame(frame)

      if utils.is_letter_q_pressed():
        collect_signal_landmarks(signal)
        break


def collect_signal_landmarks(signal):
  counter = 0

  while counter < dataset_size:
    frame = utils.image_frame()

    utils.show_image_frame(frame)
    landmarks = utils.hands_landmarks(frame)

    coordinates = []
    x_coordinates = []
    y_coordinates = []

    if landmarks:
      coordinates.append(signal)

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

      dataset_landmarks.append(coordinates)

    counter += 1
    utils.wait_letter_q_pressed()


collect_signals_data()
utils.destroy_capture_windows()
utils.landmarks_to_csv(dataset_landmarks)
