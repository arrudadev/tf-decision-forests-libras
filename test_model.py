import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import utils
from signals import SIGNALS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


model = tf.keras.models.load_model(utils.SAVED_MODEL_FILE)
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

while True:
  frame = utils.image_frame()
  landmarks = utils.hands_landmarks(frame)

  if landmarks:
    utils.draw_landmarks(frame, landmarks)
    coordinates = utils.landmark_coordinates(landmarks)

    if len(coordinates) == utils.NUMBER_OF_COORDINATES:
      coordinates = np.array(coordinates).reshape(1, -1)
      coordinates_scaled = scaler.transform(coordinates)
      coordinates_pca = pca.transform(coordinates_scaled)
      input_dict = {f'PC{i+1}': coordinates_pca[:, i]
                    for i in range(coordinates_pca.shape[1])}

      prediction = model.predict(input_dict)
      signal = SIGNALS[np.argmax(prediction)]
      print(signal)

  utils.show_image_frame(frame)

  if utils.is_letter_q_pressed():
    break

utils.destroy_capture_windows()
