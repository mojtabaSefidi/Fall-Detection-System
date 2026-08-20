from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from utils import CLASS_LABELS, CLASS_NAMES, plot_confusion_matrix


class Traditional_Models():

  def __init__(self,
               models,
               model_parameters,
               predictions=None,
               results=None,
               n_jobs=-1):
    self.models = models
    self.model_parameters = model_parameters
    self.predictions = {} if predictions is None else predictions
    self.results = {} if results is None else results
    self.n_jobs = n_jobs

  def __validation(self, X_train, y_train, validation_size=0.35):
    _, X_validation, _, y_validation = train_test_split(X_train, y_train, test_size=validation_size, stratify=y_train)
    return X_validation, y_validation

  def __flatten(self, feature_3d):
    return feature_3d.reshape(feature_3d.shape[0], feature_3d.shape[1]*feature_3d.shape[-1])

  def __feature_selector(self, X_train, y_train, X_test, class_weight, d=64):
    feature_selector = DecisionTreeClassifier(class_weight=class_weight).fit(X_train, y_train)
    important_features = np.argpartition(feature_selector.feature_importances_, -d)[-d:]
    return X_train[:,important_features], X_test[:,important_features]

  def __parameter_tuning(self, model, parameters, X_validation, y_validation, scoring='f1_macro'):
    optimizer = GridSearchCV(
        estimator=model, param_grid=parameters, scoring=scoring, n_jobs=self.n_jobs)
    optimizer.fit(X_validation, y_validation)
    return model.set_params(**optimizer.best_params_)

  def __train_model(self, model, X_train, y_train):
    return model.fit(X_train, y_train)

  def __evaluate(self, model, X_test, y_test, title=''):
    prediction = model.predict(X_test)
    self.results[title] = classification_report(
        y_test, prediction, labels=CLASS_LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    print()
    matrix = confusion_matrix(y_test, prediction, labels=CLASS_LABELS)
    plot_confusion_matrix(matrix, title=title, class_names=CLASS_NAMES)
    print()
    return prediction

  def pipeline(self,
               X_train,
               y_train,
               X_test,
               y_test,
               class_weight,
               validation_size=0.35,
               number_features=64,
               tuning_metric='f1_macro'):
    print('1) Reducing dimension of feature matrices...')
    X_train_flattened, X_test_flattened = self.__flatten(X_train), self.__flatten(X_test)
    print('2) Feature selection...')
    X_train_flattened, X_test_flattened = self.__feature_selector(X_train_flattened, y_train, X_test_flattened, class_weight=class_weight, d=number_features)
    print('3) Generating validation matrices for hyper-parameter tuning...')
    X_validation, y_validation = self.__validation(X_train_flattened, y_train, validation_size=validation_size)
    print('4) Train & Evaluation...')
    for model_name, model in self.models.items():
      print()
      print(f'----------------- Working on {model_name} -----------------')
      print()
      if model_name in self.model_parameters:
        model = self.__parameter_tuning(model, self.model_parameters[model_name], X_validation, y_validation, scoring=tuning_metric)
      model = self.__train_model(model, X_train_flattened, y_train)
      self.predictions[model_name] = self.__evaluate(model, X_test_flattened, y_test, title=model_name)
    return self.predictions, self.results
