from sklearn.utils import compute_class_weight
from itertools import groupby
from tqdm import tqdm
from math import sqrt
from pathlib import Path
import pandas as pd
import numpy as np
import random
import glob
import os


class DatasetProcessor():

  def __init__(self, enhanced_path=None):
    self.enhanced_path = Path(enhanced_path) if enhanced_path is not None else None
    self.__enhanced_files = None

  def get_file_name(self, path, ratio=0.8):
    path = os.fspath(path)
    if not path.endswith(os.sep):
      path += os.sep
    allfiles = []
    allFolders = sorted(glob.glob(path + "*"))
    for files in allFolders:
      allfiles.append(sorted(file_path for file_path in glob.glob(os.path.join(files, '*.txt'))
                             if os.path.basename(file_path).lower() != 'desktop.ini'))

    dataset = np.hstack(allfiles)
    dataset = [list(group) for _, group in groupby(
        dataset, key=lambda file_path: Path(file_path).stem.split('_')[0])]
    train = []
    test = []
    for data in dataset:
      if len(data) == 1:
        if random.random() >= ratio:
          test.extend(data)
        else:
          train.extend(data)

      else:
        random.shuffle(data)
        train.extend(data[:int(len(data)*ratio)])
        test.extend(data[int(len(data)*ratio):])

    return train, test

  def __read_data(self, data_path):
    data = pd.read_csv(data_path, header=None)
    data.columns = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x',
                    'MMA8451Q_y', 'MMA8451Q_z']
    data['MMA8451Q_z'] = data['MMA8451Q_z'].map(lambda x: str(x)[:-1])
    for name in data.columns :
      data[name] = data[name].astype(float)
    return data

  def __add_label(self, data_path, merge_feature=False):

    dataset = self.__read_data(data_path)
    labels = self.__get_label(data_path)
    if not np.isscalar(labels) and len(labels) != len(dataset):
      raise ValueError(
          f'Row mismatch for {os.path.basename(data_path)}: '
          f'{len(dataset)} sensor rows and {len(labels)} label rows')

    if not merge_feature:
      dataset['label'] = labels
      return dataset.to_numpy()

    else:
      new_dataset = pd.DataFrame()
      new_dataset['acc_1'] = dataset.apply(
          lambda row: sqrt((row.ADXL345_x ** 2 + row.ADXL345_y ** 2 + row.ADXL345_z ** 2)), axis=1)
      new_dataset['acc_2'] = dataset.apply(
          lambda row: sqrt((row.MMA8451Q_x ** 2 + row.MMA8451Q_y ** 2 + row.MMA8451Q_z ** 2)), axis=1)
      new_dataset['geo'] = dataset.apply(
          lambda row: sqrt((row.ITG3200_x ** 2 + row.ITG3200_y ** 2 + row.ITG3200_z ** 2)), axis=1)
      new_dataset['label'] = labels

      return np.round(new_dataset.to_numpy(), 2)

  def __get_label(self, data_path):
    label = os.path.basename(data_path)[0]
    if label =='D':
      return int(0)
    elif label =='F':
      label_path = self.__get_label_path(data_path)
      labels = pd.read_csv(label_path, header=None)
      return labels

  def __get_label_path(self, data_path):
    if self.enhanced_path is None:
      return os.fspath(data_path).replace('dataset', 'enhanced')

    if self.__enhanced_files is None:
      self.__enhanced_files = {}
      prefix = 'SisfallEnhanced_'
      for label_path in self.enhanced_path.rglob('*.txt'):
        stem = label_path.stem
        if stem.startswith(prefix):
          stem = stem[len(prefix):]
        if stem in self.__enhanced_files:
          raise ValueError(f'Duplicate enhanced label file for {stem}')
        self.__enhanced_files[stem] = label_path

    trial_stem = Path(data_path).stem
    if trial_stem not in self.__enhanced_files:
      raise FileNotFoundError(
          f'No enhanced label file found for {trial_stem} in {self.enhanced_path}')
    return self.__enhanced_files[trial_stem]

  def datasets_to_nparray(self, datasets_address_array, outputsize=20000000, column_dimension=10):
    result = np.zeros((outputsize, column_dimension), 'int16')
    first_index = 0
    for address in tqdm(datasets_address_array, ncols=50):
      feature = self.__add_label(address)
      result[first_index : (first_index+len(feature))] = feature
      first_index += len(feature)

    return result[:first_index]

  def windowing2d(self, dataset, window_size=200):
    window = window_size * (dataset.shape[1]-1)
    cut = dataset.shape[0] % window_size
    usable_size = dataset.shape[0] - cut
    feature = dataset[:usable_size,0:-1]
    label = dataset[:usable_size,-1]
    feature = feature.ravel().reshape(feature.size//window,window)
    label = label.reshape(label.size// window_size, window_size)
    label = label.sum(axis=1)
    label[label > 0] = 1
    feature = np.roll((np.roll(feature, -1, axis=0) - feature), 1, axis=0)
    feature[0] = 0
    return feature, label.ravel()

  def windowing3d(self, dataset, window_size=200):
    n_windows = len(dataset) // window_size
    cut = dataset.shape[0] % window_size
    usable_size = dataset.shape[0] - cut
    feature = dataset[:usable_size,0:-1]
    label = dataset[:usable_size,-1]
    feature = feature.reshape(n_windows, window_size, dataset.shape[1]-1)
    label = label.reshape(n_windows, window_size, 1)
    label = label.sum(axis=1)
    label[label > 0] = 1
    feature = np.roll((np.roll(feature, -1, axis=0) - feature), 1, axis=0)
    feature[0] = 0
    return feature, label.ravel()

  def normalizer(self, scaler, X_train, X_test):
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_test

  def generate_class_weight(self, label):
    class_weights = compute_class_weight(class_weight = "balanced",
                                         classes = np.unique(label),
                                         y = label)
    return dict(zip(np.unique(label), class_weights))
