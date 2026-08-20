"""Standard, untuned model baselines for binary fall-event detection."""

from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


def build_traditional_baselines(random_state=42, n_jobs=-1):
  """Return a compact baseline set with fixed, documented defaults."""
  return {
      'Majority baseline': DummyClassifier(strategy='most_frequent'),
      'Linear logistic': SGDClassifier(
          loss='log_loss', class_weight='balanced', max_iter=1000,
          tol=1e-3, random_state=random_state),
      'Random forest': RandomForestClassifier(
          n_estimators=100, max_depth=20, min_samples_leaf=2,
          class_weight='balanced_subsample', n_jobs=n_jobs,
          random_state=random_state),
  }


def train_traditional_baselines(models, X_train, y_train, X_test):
  """Fit fixed traditional baselines on flattened raw sensor windows."""
  X_train_flat = X_train.reshape(len(X_train), -1)
  X_test_flat = X_test.reshape(len(X_test), -1)
  fitted_models = {}
  predictions = {}
  probabilities = {}
  performance = {}

  for model_name, model in models.items():
    training_started = time.perf_counter()
    model.fit(X_train_flat, y_train)
    training_seconds = time.perf_counter() - training_started

    inference_started = time.perf_counter()
    prediction = model.predict(X_test_flat).astype(np.int8, copy=False)
    if hasattr(model, 'predict_proba'):
      probability_matrix = model.predict_proba(X_test_flat)
      positive_index = list(model.classes_).index(1)
      probability = probability_matrix[:, positive_index]
    else:
      decision = model.decision_function(X_test_flat)
      probability = 1 / (1 + np.exp(-np.clip(decision, -50, 50)))
    inference_seconds = time.perf_counter() - inference_started

    fitted_models[model_name] = model
    predictions[model_name] = prediction
    probabilities[model_name] = np.asarray(probability, dtype=np.float32)
    performance[model_name] = {
        'training_seconds': training_seconds,
        'inference_seconds': inference_seconds,
        'inference_ms_per_window': (
            1000 * inference_seconds / max(len(X_test), 1)),
        'model_size_bytes': len(pickle.dumps(
            model, protocol=pickle.HIGHEST_PROTOCOL)),
    }

  return fitted_models, predictions, probabilities, performance


def build_binary_cnn(input_shape):
  """Build a small 1D CNN for multichannel sensor windows."""
  from tensorflow import keras

  inputs = keras.layers.Input(shape=input_shape, name='sensor_window')
  x = keras.layers.Conv1D(32, 5, padding='same', use_bias=False)(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling1D(2)(x)
  x = keras.layers.Conv1D(64, 5, padding='same', use_bias=False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling1D()(x)
  x = keras.layers.Dense(64, activation='relu')(x)
  x = keras.layers.Dropout(0.30)(x)
  outputs = keras.layers.Dense(1, activation='sigmoid', name='event_probability')(x)
  return keras.Model(inputs, outputs, name='binary_sensor_cnn')


def train_binary_cnn(model, X_fit, y_fit, X_validation, y_validation,
                     class_weight, epochs=30, batch_size=128):
  """Compile and train the CNN with standard binary objectives."""
  from tensorflow import keras

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='roc_auc', curve='ROC'),
          keras.metrics.AUC(name='pr_auc', curve='PR'),
      ])
  callbacks = [
      keras.callbacks.EarlyStopping(
          monitor='val_pr_auc', mode='max', patience=5,
          restore_best_weights=True),
  ]
  started = time.perf_counter()
  history = model.fit(
      X_fit, y_fit, validation_data=(X_validation, y_validation),
      class_weight=class_weight, epochs=epochs, batch_size=batch_size,
      callbacks=callbacks, shuffle=True, verbose=2)
  return history, time.perf_counter() - started


def predict_binary_cnn(model, X_test, threshold=0.5, batch_size=128,
                       training_seconds=None):
  """Return binary predictions, positive scores, and runtime metadata."""
  # Warm-up keeps graph tracing time out of the inference measurement.
  model.predict(X_test[:min(batch_size, len(X_test))], verbose=0)
  started = time.perf_counter()
  probabilities = model.predict(
      X_test, batch_size=batch_size, verbose=0).reshape(-1)
  inference_seconds = time.perf_counter() - started
  predictions = (probabilities >= threshold).astype(np.int8)
  performance = {
      'training_seconds': training_seconds,
      'inference_seconds': inference_seconds,
      'inference_ms_per_window': (
          1000 * inference_seconds / max(len(X_test), 1)),
      'model_size_bytes': int(sum(
          np.asarray(weight).nbytes for weight in model.get_weights())),
      'parameter_count': int(model.count_params()),
  }
  return predictions, probabilities.astype(np.float32), performance


def save_baseline_models(models, cnn_model, output_path):
  """Persist fitted models in their native formats."""
  output_path = Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  for model_name, model in models.items():
    safe_name = model_name.lower().replace(' ', '_')
    with open(output_path / f'{safe_name}.pkl', 'wb') as model_file:
      pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
  cnn_model.save(output_path / 'cnn.keras')
