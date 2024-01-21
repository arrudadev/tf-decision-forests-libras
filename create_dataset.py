import time
import utils
from signals import SIGNALS

DATASET_SIZE = 2000

utils.create_dir_if_not_exists(utils.IMAGES_DATASET_DIR)

for signal in SIGNALS:
  print(f'Collecting images from the signal of {signal}')
  title = f'Press "Q" to start and make the signal of "{signal}"'

  while True:
    frame = utils.image_frame()
    landmarks = utils.hands_landmarks(frame)

    if landmarks:
      utils.draw_landmarks(frame, landmarks)

    utils.draw_title(frame, title)
    utils.show_image_frame(frame)

    if utils.is_letter_q_pressed():
      break

  counter = 0
  time.sleep(3)

  while counter < DATASET_SIZE:
    frame = utils.image_frame()
    landmarks = utils.hands_landmarks(frame)

    if landmarks:
      coordinates = utils.landmark_coordinates(landmarks)
      if len(coordinates) == utils.NUMBER_OF_COORDINATES:
        dirname = signal
        filename = f'{signal}_{counter}.jpg'
        utils.utils_save_image(dirname, filename, frame)
        print(f'Image - {filename} saved!')
        counter += 1

    utils.show_image_frame(frame)
    utils.wait_letter_q_pressed()

utils.destroy_capture_windows()
print('Done!')
