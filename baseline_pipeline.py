"""Data preparation for the reproducible binary baseline experiment."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_processor import BINARY_EVENT_LABELS, DatasetProcessor, RAW_WINDOWS
from splitting_protocols import (CHRONOLOGICAL_TRIAL_PROTOCOL,
                                 create_split_manifest,
                                 create_validation_manifest,
                                 paths_from_manifest,
                                 validate_split_manifest,
                                 validate_validation_manifest,
                                 window_indices_from_validation_manifest)


CACHE_VERSION = 'binary-baseline-v1'


def _cache_files(cache_path):
  return {
      'X_train': cache_path / 'X_train.npy',
      'y_train': cache_path / 'y_train.npy',
      'X_test': cache_path / 'X_test.npy',
      'y_test': cache_path / 'y_test.npy',
      'train_metadata': cache_path / 'train_window_metadata.csv',
      'test_metadata': cache_path / 'test_window_metadata.csv',
      'scaler': cache_path / 'channel_scaler.pkl',
      'split_manifest': cache_path / 'split_manifest.csv',
      'validation_manifest': cache_path / 'validation_manifest.csv',
  }


def _transform_channels(scaler, features):
  shape = features.shape
  transformed = scaler.transform(features.reshape(-1, shape[-1]))
  return transformed.reshape(shape).astype(np.float32, copy=False)


def prepare_binary_baseline(dataset_index, enhanced_path, cache_root,
                            window_size=200, train_ratio=0.8, seed=42,
                            reprocess=False, cache_enabled=False):
  """Split trials, create binary raw windows, and scale each sensor channel.

  Scaling statistics are fitted only on fit-trial samples. The same nine means
  and standard deviations are then applied to validation and test samples.
  """
  cache_path = (Path(cache_root) / CHRONOLOGICAL_TRIAL_PROTOCOL /
                f'{CACHE_VERSION}-w{window_size}-train{train_ratio:.2f}-seed{seed}')
  files = _cache_files(cache_path)
  cache_available = (cache_enabled and
                     all(path.exists() for path in files.values()))

  if cache_available and not reprocess:
    split_manifest = pd.read_csv(files['split_manifest'])
    validation_manifest = pd.read_csv(files['validation_manifest'])
    validate_split_manifest(
        split_manifest, require_files=False,
        expected_protocol=CHRONOLOGICAL_TRIAL_PROTOCOL,
        expected_seed=seed)
    validate_validation_manifest(validation_manifest, expected_seed=seed + 1)
    cached_trials = set(split_manifest['trial_stem'])
    current_trials = set(dataset_index['trial_stem'])
    if cached_trials != current_trials:
      raise ValueError(
          'Cached trials do not match the current paired dataset; set '
          'REPROCESS_DATA = True')
    data = {
        'X_train': np.load(files['X_train']),
        'y_train': np.load(files['y_train']),
        'X_test': np.load(files['X_test']),
        'y_test': np.load(files['y_test']),
        'train_metadata': pd.read_csv(files['train_metadata']),
        'test_metadata': pd.read_csv(files['test_metadata']),
        'scaler': _load_pickle(files['scaler']),
        'split_manifest': split_manifest,
        'validation_manifest': validation_manifest,
        'cache_path': cache_path,
        'loaded_from_cache': True,
    }
  else:
    split_manifest = create_split_manifest(
        dataset_index, protocol=CHRONOLOGICAL_TRIAL_PROTOCOL,
        train_ratio=train_ratio, seed=seed)
    validation_manifest = create_validation_manifest(
        split_manifest, train_ratio=train_ratio, seed=seed + 1)
    train_paths, test_paths = paths_from_manifest(split_manifest)
    processor = DatasetProcessor(enhanced_path=enhanced_path)
    X_train, y_train, train_metadata = processor.process_trials(
        train_paths, window_size=window_size, preprocessing_mode=RAW_WINDOWS,
        label_mode=BINARY_EVENT_LABELS)
    X_test, y_test, test_metadata = processor.process_trials(
        test_paths, window_size=window_size, preprocessing_mode=RAW_WINDOWS,
        label_mode=BINARY_EVENT_LABELS)
    fit_indices, _ = window_indices_from_validation_manifest(
        train_metadata, validation_manifest)

    scaler = StandardScaler()
    scaler.fit(X_train[fit_indices].reshape(-1, X_train.shape[-1]))
    X_train = _transform_channels(scaler, X_train)
    X_test = _transform_channels(scaler, X_test)

    if cache_enabled:
      cache_path.mkdir(parents=True, exist_ok=True)
      np.save(files['X_train'], X_train)
      np.save(files['y_train'], y_train)
      np.save(files['X_test'], X_test)
      np.save(files['y_test'], y_test)
      train_metadata.to_csv(files['train_metadata'], index=False)
      test_metadata.to_csv(files['test_metadata'], index=False)
      split_manifest.to_csv(files['split_manifest'], index=False)
      validation_manifest.to_csv(files['validation_manifest'], index=False)
      _save_pickle(scaler, files['scaler'])
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_metadata': train_metadata,
        'test_metadata': test_metadata,
        'scaler': scaler,
        'split_manifest': split_manifest,
        'validation_manifest': validation_manifest,
        'cache_path': cache_path,
        'loaded_from_cache': False,
    }

  fit_indices, validation_indices = window_indices_from_validation_manifest(
      data['train_metadata'], data['validation_manifest'])
  processor = DatasetProcessor(enhanced_path=enhanced_path)
  processor.validate_model_inputs(
      data['X_train'], data['y_train'], data['train_metadata'],
      expected_labels=(0, 1))
  processor.validate_model_inputs(
      data['X_test'], data['y_test'], data['test_metadata'],
      expected_labels=(0, 1))
  data['fit_indices'] = fit_indices
  data['validation_indices'] = validation_indices
  return data


def _save_pickle(value, path):
  with Path(path).open('wb') as output_file:
    pickle.dump(value, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path):
  with Path(path).open('rb') as input_file:
    return pickle.load(input_file)


def window_split_summary(y_train, y_test, fit_indices, validation_indices):
  rows = []
  for name, labels in (
      ('Fit', y_train[fit_indices]),
      ('Validation', y_train[validation_indices]),
      ('Train total', y_train),
      ('Test', y_test),
  ):
    positives = int(np.count_nonzero(labels == 1))
    rows.append({
        'split': name,
        'windows': len(labels),
        'negative_windows': int(np.count_nonzero(labels == 0)),
        'positive_windows': positives,
        'positive_rate': positives / len(labels),
    })
  return pd.DataFrame(rows)
