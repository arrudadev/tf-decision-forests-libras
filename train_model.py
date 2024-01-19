import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

print(features_reduced_df.head())

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
    features_reduced_df, label="signal")

model = tfdf.keras.RandomForestModel()
model.fit(train_dataset)
model.save(utils.SAVED_MODEL_FILE)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
