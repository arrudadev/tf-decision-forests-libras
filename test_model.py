import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import utils
from signals import SIGNALS

model = tf.keras.models.load_model(utils.SAVED_MODEL_FILE)

while True:
  frame = utils.image_frame()
  landmarks = utils.hands_landmarks(frame)

  if landmarks:
    utils.draw_landmarks(frame, landmarks)
    coordinates = utils.landmark_coordinates(landmarks)

    prediction = model.predict([np.asarray(coordinates)])
    print(prediction)

  utils.show_image_frame(frame)

  if utils.is_letter_q_pressed():
    break

utils.destroy_capture_windows()
