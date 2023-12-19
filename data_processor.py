from sklearn.utils import compute_class_weight
from itertools import groupby
from tqdm import tqdm
from math import sqrt
import pandas as pd
import numpy as np
import random
import glob
import os


class DatasetProcessor():

  def get_file_name(self, path, ratio=0.8):
    allfiles = []
    allFolders = sorted(glob.glob(path + "*"))
    for files in allFolders:
      allfiles.append(sorted(glob.glob(files+"/*.txt")))
    if 'desktop.ini' in allfiles:
          allfiles.remove('desktop.ini')

    dataset = np.hstack(allfiles)
    start = dataset[0].rfind('/') + 1
    end = dataset[0][start:].find('_') + start
    dataset = [list(g) for k, g in groupby(dataset, key=lambda x: x[start:end])]
    train = []
    test = []
    for data in dataset:
      if len(data) == 1:
        if random.randint(1,100)>=ratio:
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

    if not merge_feature:
      dataset['label'] = self.__get_label(data_path)
      return dataset.to_numpy()

    else:
      new_dataset = pd.DataFrame()
      new_dataset['acc_1'] = dataset.apply(
          lambda row: sqrt((row.ADXL345_x ** 2 + row.ADXL345_y ** 2 + row.ADXL345_z ** 2)), axis=1)
      new_dataset['acc_2'] = dataset.apply(
          lambda row: sqrt((row.MMA8451Q_x ** 2 + row.MMA8451Q_y ** 2 + row.MMA8451Q_z ** 2)), axis=1)
      new_dataset['geo'] = dataset.apply(
          lambda row: sqrt((row.ITG3200_x ** 2 + row.ITG3200_y ** 2 + row.ITG3200_z ** 2)), axis=1)
      new_dataset['label'] = self.__get_label(data_path)

      return np.round(new_dataset.to_numpy(), 2)

  def __get_label(self, data_path):
    label = data_path[54]
    if label =='D':
      return int(0)
    elif label =='F':
      label_path = data_path.replace('dataset', 'enhanced')
      labels = pd.read_csv(label_path, header=None)
      return labels

  def datasets_to_nparray(self, datasets_address_array, outputsize=20000000, column_dimension=10):
    result = np.zeros((outputsize, column_dimension), 'int16')
    first_index = 0
    for address in tqdm(datasets_address_array, ncols=50):
      feature = self.__add_label(address)
      result[first_index : (first_index+len(feature))] = feature
      first_index += len(feature)

    return result[result.sum(axis=1) != 0]

  def windowing2d(self, dataset, window_size=200):
    window = window_size * (dataset.shape[1]-1)
    cut = dataset.shape[0] % window_size
    feature = dataset[:-cut,0:-1]
    label = dataset[:-cut,-1]
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
    feature = dataset[:-cut,0:-1]
    label = dataset[:-cut,-1]
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
