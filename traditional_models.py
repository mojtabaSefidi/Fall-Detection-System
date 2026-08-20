from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import numpy as np
import pickle
import time

from utils import CLASS_LABELS, CLASS_NAMES


class Traditional_Models():

  def __init__(self,
               models,
               model_parameters,
               predictions=None,
               results=None,
               probabilities=None,
               performance=None,
               n_jobs=-1,
               random_state=42):
    self.models = models
    self.model_parameters = model_parameters
    self.predictions = {} if predictions is None else predictions
    self.results = {} if results is None else results
    self.probabilities = {} if probabilities is None else probabilities
    self.performance = {} if performance is None else performance
    self.n_jobs = n_jobs
    self.random_state = random_state

  def __validation(self, X_train, y_train, validation_size=0.35):
    _, X_validation, _, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size, stratify=y_train,
        random_state=self.random_state)
    return X_validation, y_validation

  def __flatten(self, feature_3d):
    return feature_3d.reshape(feature_3d.shape[0], feature_3d.shape[1]*feature_3d.shape[-1])

  def __feature_selector(self, X_selector, y_selector, X_train, X_test,
                         class_weight, d=64,
                         feature_indices_path=None):
    d = min(d, X_train.shape[1])
    feature_indices_path = (Path(feature_indices_path)
                            if feature_indices_path is not None else None)
    if feature_indices_path is not None and feature_indices_path.exists():
      important_features = np.load(feature_indices_path)
      if (important_features.ndim != 1 or len(important_features) != d or
          important_features.max(initial=-1) >= X_train.shape[1]):
        raise ValueError('Cached feature indices do not match the current dataset')
      print(f'   Loaded {d} cached feature indices')
    else:
      feature_selector = DecisionTreeClassifier(
          class_weight=class_weight,
          random_state=self.random_state).fit(X_selector, y_selector)
      important_features = np.argsort(feature_selector.feature_importances_)[-d:]
      if feature_indices_path is not None:
        feature_indices_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_indices_path, important_features)
    return (X_train[:, important_features], X_test[:, important_features],
            important_features)

  def __parameter_tuning(self, model, parameters, X_validation, y_validation, scoring='f1_macro'):
    optimizer = GridSearchCV(
        estimator=model, param_grid=parameters, scoring=scoring, n_jobs=self.n_jobs)
    optimizer.fit(X_validation, y_validation)
    return model.set_params(**optimizer.best_params_), optimizer.best_params_

  def __train_model(self, model, X_train, y_train):
    return model.fit(X_train, y_train)

  def __evaluate(self, prediction, y_test, title=''):
    self.results[title] = classification_report(
        y_test, prediction, labels=CLASS_LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)

  def pipeline(self,
               X_train,
               y_train,
               X_test,
               y_test,
               class_weight,
               validation_size=0.35,
               number_features=64,
               tuning_metric='f1_macro',
               feature_indices_path=None,
               fit_indices=None,
               validation_indices=None):
    print('1) Reducing dimension of feature matrices...')
    X_train_flattened, X_test_flattened = self.__flatten(X_train), self.__flatten(X_test)
    print('2) Feature selection...')
    selector_X = (X_train_flattened if fit_indices is None
                  else X_train_flattened[fit_indices])
    selector_y = y_train if fit_indices is None else y_train[fit_indices]
    X_train_flattened, X_test_flattened, important_features = self.__feature_selector(
        selector_X, selector_y, X_train_flattened, X_test_flattened,
        class_weight=class_weight, d=number_features,
        feature_indices_path=feature_indices_path)
    print('3) Generating validation matrices for hyper-parameter tuning...')
    if validation_indices is None:
      X_validation, y_validation = self.__validation(
          X_train_flattened, y_train, validation_size=validation_size)
    else:
      X_validation = X_train_flattened[validation_indices]
      y_validation = y_train[validation_indices]
    print('4) Train & Evaluation...')
    for model_name, model in self.models.items():
      print()
      print(f'----------------- Working on {model_name} -----------------')
      print()
      training_started = time.perf_counter()
      best_parameters = {}
      if model_name in self.model_parameters:
        model, best_parameters = self.__parameter_tuning(
            model, self.model_parameters[model_name], X_validation, y_validation,
            scoring=tuning_metric)
      model = self.__train_model(model, X_train_flattened, y_train)
      training_seconds = time.perf_counter() - training_started

      inference_started = time.perf_counter()
      probabilities = model.predict_proba(X_test_flattened)
      prediction = model.classes_[np.argmax(probabilities, axis=1)]
      inference_seconds = time.perf_counter() - inference_started

      self.predictions[model_name] = prediction
      self.probabilities[model_name] = probabilities
      self.__evaluate(prediction, y_test, title=model_name)
      self.performance[model_name] = {
          'training_seconds': training_seconds,
          'inference_seconds': inference_seconds,
          'inference_ms_per_window': 1000 * inference_seconds / max(len(X_test), 1),
          'model_size_bytes': len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
          'selected_feature_count': len(important_features),
          'probability_classes': model.classes_.tolist(),
          'best_parameters': best_parameters,
      }
    return self.predictions, self.results
