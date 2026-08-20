from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix

from utils import (CLASS_LABELS, CLASS_NAMES, plot_auc_curve,
                   plot_confusion_matrix, plot_history, plot_metrics,
                   plot_precision_recall_curve, plot_roc_curve)

class Train_Evaluate_Deep():

  def __init__(self,
               predictions=None,
               results=None,
               probabilities=None,
               performance=None):

      self.predictions = {} if predictions is None else predictions
      self.results = {} if results is None else results
      self.probabilities = {} if probabilities is None else probabilities
      self.performance = {} if performance is None else performance

  def build_cnn(self, input_size, units=128, drop_rate=0.25, filter=32, kernel_size=(1*9), output_size=3):
    input = keras.layers.Input((input_size))
    x = keras.layers.Conv1D(filters=filter//2, kernel_size=kernel_size, padding='same', activation='relu', name="conv1")(input)
    x = keras.layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu', name="conv2")(x)
    x = keras.layers.Conv1D(filters=filter*2, kernel_size=kernel_size, padding='same', activation='relu', name="conv3")(x)
    x = tf.keras.layers.Flatten()(x)
    classifier = keras.layers.Dense(units*4, activation='relu')(x)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(units, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(units//2, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    output = keras.layers.Dense(output_size, activation='softmax')(classifier)
    model = keras.Model(inputs=input, outputs=output)
    return model

  def build_lstm(self, input_size, units=128, drop_rate=0.25, lstm_units=16, output_size=3):
    input = keras.layers.Input((input_size))
    x = keras.layers.LSTM(units=lstm_units//2, input_shape=input_size, return_sequences=True, name="lstm1")(input)
    x = keras.layers.LSTM(units=lstm_units, input_shape=input_size, return_sequences=True, name="lstm2")(x)
    x = keras.layers.LSTM(units=lstm_units*2, input_shape=input_size, return_sequences=True, name="lstm3")(x)
    x = tf.keras.layers.Flatten()(x)
    classifier = keras.layers.Dense(units*4, activation='relu')(x)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(units, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(units//2, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    output = keras.layers.Dense(output_size, activation='softmax')(classifier)
    model = keras.Model(inputs=input, outputs=output)
    return model

  def build_mlp(self, input_size, hidden_layer_size=128, output_size=3, drop_rate=0.25):
    input = keras.layers.Input((input_size))
    x = keras.layers.Dense(hidden_layer_size//4, activation='relu')(input)
    x = keras.layers.Dense(hidden_layer_size//2, activation='relu')(x)
    x = keras.layers.Dense(hidden_layer_size, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    classifier = keras.layers.Dense(hidden_layer_size*4, activation='relu')(x)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(hidden_layer_size, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    classifier = keras.layers.Dense(hidden_layer_size//2, activation='relu')(classifier)
    classifier = keras.layers.Dropout(drop_rate)(classifier)
    output = keras.layers.Dense(output_size, activation='softmax')(classifier)
    model = keras.Model(inputs=input, outputs=output)
    return model

  def build_AE(self, input_size, latent_dim=128, filter=64, kernel_size=(1*9)):
    input = keras.layers.Input((input_size))
    encoder = keras.layers.Conv1D(filters=filter//2, kernel_size=kernel_size, padding='same', activation='relu', name="conv1")(input)
    encoder = keras.layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same', activation='relu', name="conv2")(encoder)
    encoder = tf.keras.layers.Flatten()(encoder)
    encoder = keras.layers.Dense(latent_dim*4, activation='relu')(encoder)
    encoder = keras.layers.Dense(latent_dim*2, activation='relu')(encoder)

    latent = keras.layers.Dense(latent_dim, activation='relu')(encoder)

    decoder = keras.layers.Dense(latent_dim*2, activation='relu')(latent)
    decoder = keras.layers.Dense(latent_dim*4, activation='relu')(decoder)
    decoder = keras.layers.Dense(filter*input_size[0], activation='relu')(decoder)
    decoder = keras.layers.Reshape((input_size[0], filter))(decoder)
    decoder = keras.layers.Conv1DTranspose(filters=filter, kernel_size=kernel_size, padding='same', activation='relu')(decoder)
    decoder = keras.layers.Conv1DTranspose(filters=filter//2, kernel_size=kernel_size, padding='same', activation='relu')(decoder)

    output = keras.layers.Conv1DTranspose(filters=input_size[-1], kernel_size=kernel_size, padding='same', activation='relu')(decoder)
    model = keras.Model(inputs=input, outputs=output)
    return model

  def train_deep_model(self,
                       model,
                       X_train,
                       y_train,
                       metrics,
                       loss_function,
                       optimizer,
                       callbacks,
                       class_weight=None,
                       epochs=100,
                       batch_size=128,
                       validation_split=0.2,
                       validation_data=None,
                       verbose=2):

    print('----------------------------------')
    model.summary()
    print('----------------------------------')

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)

    fit_arguments = {
        'batch_size': batch_size,
        'epochs': epochs,
        'shuffle': True,
        'callbacks': callbacks,
        'verbose': verbose,
    }
    if validation_data is None:
      fit_arguments['validation_split'] = validation_split
    else:
      fit_arguments['validation_data'] = validation_data
    if class_weight is not None:
      fit_arguments['class_weight'] = class_weight
    return model.fit(X_train, y_train, **fit_arguments)

  def evaluate(self, model, X_test, y_test, batch_size=128, title='', model_name='',
               plot=True, training_seconds=None):
    model.predict(X_test[:min(batch_size, len(X_test))], batch_size=batch_size,
                  verbose=0)
    inference_started = time.perf_counter()
    prediction_scores = model.predict(X_test, batch_size=batch_size, verbose=0)
    inference_seconds = time.perf_counter() - inference_started
    prediction = np.argmax(prediction_scores, axis=1)
    self.predictions[model_name] = prediction
    self.probabilities[model_name] = prediction_scores
    if plot:
      print()
      print('1) Plot ROC Curve...')
      print()
      plot_roc_curve(y_test, prediction_scores, title='ROC Curve of {model_name} Model'.format(model_name=model_name), model_name=model_name, file_name=None)
      print()
      print('2) Plot AUC Curve...')
      print()
      plot_auc_curve(y_test, prediction_scores, title='AUC Curve of {model_name} Model'.format(model_name=model_name), model_name=model_name, file_name = None)
      print()
      print('3) Plot Percision_Recall Curve......')
      print()
      plot_precision_recall_curve(y_test, prediction_scores, title='Percision_Recall Curve of {model_name} Model'.format(model_name=model_name), model_name=model_name, file_name=None)
    report = classification_report(
        y_test, prediction, labels=CLASS_LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    self.results[model_name] = report
    self.performance[model_name] = {
        'training_seconds': training_seconds,
        'inference_seconds': inference_seconds,
        'inference_ms_per_window': 1000 * inference_seconds / max(len(X_test), 1),
        'model_size_bytes': int(sum(np.asarray(weight).nbytes for weight in model.get_weights())),
        'parameter_count': int(model.count_params()),
        'probability_classes': [0, 1, 2],
    }
    if plot:
      print()
      matrix = confusion_matrix(y_test, prediction, labels=CLASS_LABELS)
      plot_confusion_matrix(matrix, title=title, class_names=CLASS_NAMES)
      print()
    return prediction

  def plot_learning_curves(self, history):
    print('1) Plot learning process based on different metrics...')
    print()
    plot_metrics(history)
    print()
    print('2) Plot learning curve...')
    print()
    return plot_history(history)
