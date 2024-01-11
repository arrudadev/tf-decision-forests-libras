import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflowjs as tfjs
import utils


model = tf.keras.models.load_model(utils.SAVED_MODEL_FILE)

tfjs.converters.tf_saved_model_conversion_v2.convert_keras_model_to_graph_model(
    model, utils.CONVERTED_MODEL_FILE)
