import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

dataset_data_frame = pd.read_csv("./dataset_landmarks.csv")
train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
    dataset_data_frame, label="signal")

model = tfdf.keras.GradientBoostedTreesModel()
model.fit(train_dataset)
model.save("./saved_model")
