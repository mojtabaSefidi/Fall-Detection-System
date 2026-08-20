from sklearn.utils import compute_class_weight
from itertools import groupby
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import random
import glob
import os
import re


RAW_WINDOWS = 'raw_windows'
PREVIOUS_WINDOW_DIFFERENCE = 'previous_window_difference'
SUPPORTED_PREPROCESSING_MODES = {RAW_WINDOWS, PREVIOUS_WINDOW_DIFFERENCE}
BINARY_EVENT_LABELS = 'binary_event'
THREE_CLASS_LABELS = 'three_class'
SUPPORTED_LABEL_MODES = {BINARY_EVENT_LABELS, THREE_CLASS_LABELS}


class DatasetProcessor():

  def __init__(self, enhanced_path=None):
    self.enhanced_path = Path(enhanced_path) if enhanced_path is not None else None
    self.__enhanced_files = None

  @staticmethod
  def _canonical_trial_parts(data_path):
    data_path = Path(data_path)
    parts = data_path.stem.split('_')
    if len(parts) != 3:
      raise ValueError(f'Unexpected SisFall file name: {data_path.name}')
    activity, filename_subject, trial = parts
    folder_subject = data_path.parent.name
    subject = (folder_subject
               if re.fullmatch(r'S[AE]\d{2}', folder_subject)
               else filename_subject)
    return activity, subject, trial, filename_subject

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
    if data.shape[1] != 9:
      raise ValueError(
          f'Expected 9 sensor columns in {Path(data_path).name}, found {data.shape[1]}')
    data.columns = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x',
                    'MMA8451Q_y', 'MMA8451Q_z']
    data['MMA8451Q_z'] = data['MMA8451Q_z'].astype(str).str.rstrip(';')
    return data.astype(np.float32)

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
      values = dataset.to_numpy(dtype=np.float32, copy=False)
      new_dataset = pd.DataFrame({
          'acc_1': np.linalg.norm(values[:, 0:3], axis=1),
          'acc_2': np.linalg.norm(values[:, 6:9], axis=1),
          'geo': np.linalg.norm(values[:, 3:6], axis=1),
      })
      new_dataset['label'] = labels

      return np.round(new_dataset.to_numpy(), 2)

  def __get_label(self, data_path):
    label = os.path.basename(data_path)[0]
    if label =='D':
      return int(0)
    elif label =='F':
      label_path = self.__get_label_path(data_path)
      labels = pd.read_csv(label_path, header=None).to_numpy(dtype=np.int8).ravel()
      invalid_labels = set(np.unique(labels)) - {0, 1, 2}
      if invalid_labels:
        raise ValueError(
            f'Unexpected labels {sorted(invalid_labels)} in {Path(label_path).name}')
      return labels
    raise ValueError(f'Unknown activity type in {Path(data_path).name}')

  def __get_label_path(self, data_path):
    if self.enhanced_path is None:
      return os.fspath(data_path).replace('dataset', 'enhanced')

    if self.__enhanced_files is None:
      self.__enhanced_files = {}
      prefix = 'SisfallEnhanced_'
      for label_path in self.enhanced_path.rglob('*.txt'):
        stem = label_path.stem
        is_enhanced_name = stem.startswith(prefix)
        if is_enhanced_name:
          stem = stem[len(prefix):]
        if re.fullmatch(r'[DF]\d{2}_S[AE]\d{2}_R\d{2}', stem) is None:
          continue
        if stem in self.__enhanced_files:
          existing_is_enhanced = self.__enhanced_files[stem].stem.startswith(prefix)
          if is_enhanced_name and not existing_is_enhanced:
            self.__enhanced_files[stem] = label_path
            continue
          if not is_enhanced_name and existing_is_enhanced:
            continue
          raise ValueError(f'Duplicate enhanced label file for {stem}')
        self.__enhanced_files[stem] = label_path

    activity, subject, trial, _ = self._canonical_trial_parts(data_path)
    trial_stem = '_'.join([activity, subject, trial])
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
    label, _, _ = self._three_class_window_labels(label)
    feature = np.roll((np.roll(feature, -1, axis=0) - feature), 1, axis=0)
    feature[0] = 0
    return feature, label

  def windowing3d(self, dataset, window_size=200):
    n_windows = len(dataset) // window_size
    cut = dataset.shape[0] % window_size
    usable_size = dataset.shape[0] - cut
    feature = dataset[:usable_size,0:-1]
    label = dataset[:usable_size,-1]
    feature = feature.reshape(n_windows, window_size, dataset.shape[1]-1)
    label = label.reshape(n_windows, window_size)
    label, _, _ = self._three_class_window_labels(label)
    feature = np.roll((np.roll(feature, -1, axis=0) - feature), 1, axis=0)
    feature[0] = 0
    return feature, label

  @staticmethod
  def _three_class_window_labels(label_windows):
    """Label windows as 0=no fall, 2=pre-fall, or 1=fall.

    Event windows use the dominant non-zero stage. Ties are assigned to fall,
    which avoids losing a fall event while preserving pre-fall transition windows.
    """
    pre_fall_count = np.count_nonzero(label_windows == 2, axis=1)
    fall_count = np.count_nonzero(label_windows == 1, axis=1)
    targets = np.zeros(len(label_windows), dtype=np.int8)
    targets[pre_fall_count > fall_count] = 2
    targets[(fall_count > 0) & (fall_count >= pre_fall_count)] = 1
    return targets, pre_fall_count, fall_count

  def process_trials(self, dataset_paths, window_size=200,
                     preprocessing_mode=PREVIOUS_WINDOW_DIFFERENCE,
                     label_mode=THREE_CLASS_LABELS):
    """Create windows independently inside every recording.

    Incomplete trial tails are dropped. The compatibility difference transform
    compares corresponding samples in adjacent windows and is reset at each trial
    boundary; raw_windows keeps the original samples. Metadata makes every window
    traceable to its recording and label composition.
    """
    if preprocessing_mode not in SUPPORTED_PREPROCESSING_MODES:
      raise ValueError(
          f'Unsupported preprocessing mode {preprocessing_mode!r}; choose from '
          f'{sorted(SUPPORTED_PREPROCESSING_MODES)}')
    if label_mode not in SUPPORTED_LABEL_MODES:
      raise ValueError(
          f'Unsupported label mode {label_mode!r}; choose from '
          f'{sorted(SUPPORTED_LABEL_MODES)}')
    feature_batches = []
    label_batches = []
    metadata_rows = []

    for data_path in tqdm(dataset_paths, ncols=70, desc='Windowing trials'):
      data_path = Path(data_path)
      sensor_data = self.__read_data(data_path).to_numpy(dtype=np.float32, copy=False)
      labels = self.__get_label(data_path)
      if np.isscalar(labels):
        labels = np.full(len(sensor_data), labels, dtype=np.int8)
      if len(labels) != len(sensor_data):
        raise ValueError(
            f'Row mismatch for {data_path.name}: '
            f'{len(sensor_data)} sensor rows and {len(labels)} label rows')

      n_windows = len(sensor_data) // window_size
      if n_windows == 0:
        continue
      usable_size = n_windows * window_size
      windows = sensor_data[:usable_size].reshape(n_windows, window_size, 9)
      label_windows = labels[:usable_size].reshape(n_windows, window_size)
      if label_mode == BINARY_EVENT_LABELS:
        targets = np.any(label_windows != 0, axis=1).astype(np.int8)
      else:
        targets, _, _ = self._three_class_window_labels(label_windows)
      pre_fall_count = np.count_nonzero(label_windows == 2, axis=1)
      fall_count = np.count_nonzero(label_windows == 1, axis=1)
      pre_fall_start = np.where(
          pre_fall_count > 0, np.argmax(label_windows == 2, axis=1), np.nan)
      fall_start = np.where(
          fall_count > 0, np.argmax(label_windows == 1, axis=1), np.nan)

      if preprocessing_mode == PREVIOUS_WINDOW_DIFFERENCE:
        window_differences = np.empty_like(windows, dtype=np.float32)
        window_differences[0] = 0
        np.subtract(windows[1:], windows[:-1], out=window_differences[1:])
        windows = window_differences

      activity, subject, trial, filename_subject = self._canonical_trial_parts(
          data_path)
      trial_stem = '_'.join([activity, subject, trial])
      no_fall_count = window_size - pre_fall_count - fall_count
      for window_index in range(n_windows):
        start_sample = window_index * window_size
        metadata_rows.append({
            'trial_stem': trial_stem,
            'source_trial_stem': data_path.stem,
            'filename_subject_mismatch': filename_subject != subject,
            'activity': activity,
            'subject': subject,
            'age_group': 'Adult' if subject.startswith('SA') else 'Elderly',
            'trial': trial,
            'source_file': str(data_path),
            'preprocessing_mode': preprocessing_mode,
            'label_mode': label_mode,
            'window_index': window_index,
            'start_sample': start_sample,
            'end_sample': start_sample + window_size - 1,
            'start_time_seconds': start_sample / 200,
            'end_time_seconds': (start_sample + window_size) / 200,
            'no_fall_samples': int(no_fall_count[window_index]),
            'pre_fall_samples': int(pre_fall_count[window_index]),
            'fall_samples': int(fall_count[window_index]),
            'no_fall_fraction': no_fall_count[window_index] / window_size,
            'pre_fall_fraction': pre_fall_count[window_index] / window_size,
            'fall_fraction': fall_count[window_index] / window_size,
            'pre_fall_start_fraction': pre_fall_start[window_index] / window_size,
            'fall_start_fraction': fall_start[window_index] / window_size,
            'target': int(targets[window_index]),
        })

      feature_batches.append(windows)
      label_batches.append(targets)

    if not feature_batches:
      return (np.empty((0, window_size, 9), dtype=np.float32),
              np.empty(0, dtype=np.int8),
              pd.DataFrame(metadata_rows))
    return (np.concatenate(feature_batches),
            np.concatenate(label_batches),
            pd.DataFrame(metadata_rows))

  def normalizer(self, scaler, X_train, X_test):
    """Fit nine independent channel statistics on train and transform test."""
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return (X_train.astype(np.float32, copy=False),
            X_test.astype(np.float32, copy=False))

  @staticmethod
  def validate_model_inputs(features, labels, metadata=None,
                            expected_labels=(0, 1)):
    if features.ndim != 3 or features.shape[-1] != 9:
      raise ValueError(
          f'Expected model input shaped (windows, samples, 9), found {features.shape}')
    if len(features) != len(labels):
      raise ValueError('Feature and label window counts do not match')
    if metadata is not None and len(metadata) != len(labels):
      raise ValueError('Window metadata and label counts do not match')
    observed = set(np.unique(labels))
    expected = set(expected_labels)
    if observed != expected:
      raise ValueError(
          f'Expected model labels {sorted(expected)}, found {sorted(observed)}')
    return True

  def generate_class_weight(self, label):
    class_weights = compute_class_weight(class_weight = "balanced",
                                         classes = np.unique(label),
                                         y = label)
    return dict(zip(np.unique(label), class_weights))
