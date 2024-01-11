import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import utils

dataset_data_frame = pd.read_csv(utils.DATASET_FILE_CSV)
train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
    dataset_data_frame, label="signal")

model = tfdf.keras.GradientBoostedTreesModel()
model.fit(train_dataset)
model.save(utils.SAVED_MODEL_FILE)
