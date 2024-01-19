import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import utils
import joblib

dataset_data_frame = pd.read_csv(utils.DATASET_FILE_CSV)

features = dataset_data_frame.drop('signal', axis=1)
labels = dataset_data_frame['signal']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=0.95)
features_reduced = pca.fit_transform(features_scaled)

features_reduced_df = pd.DataFrame(features_reduced, columns=[
                                   f'PC{i+1}' for i in range(features_reduced.shape[1])])
features_reduced_df['signal'] = labels.values

train_df, test_df = train_test_split(
    features_reduced_df, test_size=0.2, random_state=42)

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="signal")
test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="signal")

print(train_df.shape)
print(train_df.head())

print(test_df.shape)
print(test_df.head())

model = tfdf.keras.RandomForestModel()
model.fit(train_dataset)

predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)

true_labels = test_df['signal'].to_numpy()

accuracy = np.mean(predicted_labels == true_labels)
print(f"Precisão: {accuracy}")

model.save(utils.SAVED_MODEL_FILE)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
