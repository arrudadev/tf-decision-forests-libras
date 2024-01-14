import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import utils
from signals import SIGNALS

NUMBER_OF_COORDINATES = 42

model = tf.keras.models.load_model(utils.SAVED_MODEL_FILE)

while True:
  frame = utils.image_frame()
  landmarks = utils.hands_landmarks(frame)

  if landmarks:
    utils.draw_landmarks(frame, landmarks)
    coordinates = utils.landmark_coordinates(landmarks)

    if len(coordinates) == NUMBER_OF_COORDINATES:
      input_data = utils.coordinates_to_input_data(coordinates)
      prediction = model.predict(input_data)
      signal = SIGNALS[np.argmax(prediction)]
      print(signal)

  utils.show_image_frame(frame)

  if utils.is_letter_q_pressed():
    break

utils.destroy_capture_windows()
