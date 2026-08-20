from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SAMPLING_RATE = 200
SENSOR_COLUMNS = [
    'ADXL345_x', 'ADXL345_y', 'ADXL345_z',
    'ITG3200_x', 'ITG3200_y', 'ITG3200_z',
    'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z',
]

ACTIVITIES = {
    'D01': ('Walking slowly', 1, 100),
    'D02': ('Walking quickly', 1, 100),
    'D03': ('Jogging slowly', 1, 100),
    'D04': ('Jogging quickly', 1, 100),
    'D05': ('Walking upstairs and downstairs slowly', 5, 25),
    'D06': ('Walking upstairs and downstairs quickly', 5, 25),
    'D07': ('Slowly sit in a half-height chair and get up', 5, 12),
    'D08': ('Quickly sit in a half-height chair and get up', 5, 12),
    'D09': ('Slowly sit in a low-height chair and get up', 5, 12),
    'D10': ('Quickly sit in a low-height chair and get up', 5, 12),
    'D11': ('Try to get up and collapse into a chair', 5, 12),
    'D12': ('Lie down slowly and sit again', 5, 12),
    'D13': ('Lie down quickly and sit again', 5, 12),
    'D14': ('Change between back and lateral lying positions', 5, 12),
    'D15': ('Bend slowly at the knees and get up', 5, 12),
    'D16': ('Bend slowly without bending the knees and get up', 5, 12),
    'D17': ('Get into and out of a car', 5, 25),
    'D18': ('Stumble while walking', 5, 12),
    'D19': ('Gently jump without falling', 5, 12),
    'F01': ('Forward fall while walking caused by a slip', 5, 15),
    'F02': ('Backward fall while walking caused by a slip', 5, 15),
    'F03': ('Lateral fall while walking caused by a slip', 5, 15),
    'F04': ('Forward fall while walking caused by a trip', 5, 15),
    'F05': ('Forward fall while jogging caused by a trip', 5, 15),
    'F06': ('Vertical fall while walking caused by fainting', 5, 15),
    'F07': ('Fall dampened with hands on a table', 5, 15),
    'F08': ('Forward fall while trying to get up', 5, 15),
    'F09': ('Lateral fall while trying to get up', 5, 15),
    'F10': ('Forward fall while trying to sit down', 5, 15),
    'F11': ('Backward fall while trying to sit down', 5, 15),
    'F12': ('Lateral fall while trying to sit down', 5, 15),
    'F13': ('Forward fall while sitting', 5, 15),
    'F14': ('Backward fall while sitting', 5, 15),
    'F15': ('Lateral fall while sitting', 5, 15),
}

LABEL_NAMES = {0: 'No fall', 2: 'Pre-fall', 1: 'Fall'}
LABEL_COLORS = {0: '#b7e4c7', 2: '#ffd166', 1: '#ef476f'}
TRIAL_PATTERN = re.compile(
    r'^(?P<activity>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})$')


def normalize_trial_stem(path):
  stem = Path(path).stem
  prefix = 'SisfallEnhanced_'
  if stem.startswith(prefix):
    stem = stem[len(prefix):]
  return stem


def parse_trial_name(path):
  stem = normalize_trial_stem(path)
  match = TRIAL_PATTERN.fullmatch(stem)
  if match is None:
    raise ValueError(f'Invalid SisFall trial name: {Path(path).name}')
  metadata = match.groupdict()
  activity = metadata['activity']
  if activity not in ACTIVITIES:
    raise ValueError(f'Unknown SisFall activity code: {activity}')
  description, expected_trials, duration_seconds = ACTIVITIES[activity]
  metadata.update({
      'trial_stem': stem,
      'activity_type': 'ADL' if activity.startswith('D') else 'Fall',
      'activity_description': description,
      'expected_trials': expected_trials,
      'expected_duration_seconds': duration_seconds,
      'age_group': 'Adult' if metadata['subject'].startswith('SA') else 'Elderly',
  })
  return metadata


def parse_sensor_trial_path(path):
  """Parse a sensor path and make the subject folder authoritative."""
  path = Path(path)
  metadata = parse_trial_name(path)
  metadata['source_trial_stem'] = metadata['trial_stem']
  folder_subject = path.parent.name
  folder_is_subject = re.fullmatch(r'S[AE]\d{2}', folder_subject) is not None
  mismatch = folder_is_subject and folder_subject != metadata['subject']
  metadata['filename_subject_mismatch'] = mismatch
  if mismatch:
    metadata['subject'] = folder_subject
    metadata['age_group'] = 'Adult' if folder_subject.startswith('SA') else 'Elderly'
    metadata['trial_stem'] = '_'.join([
        metadata['activity'], folder_subject, metadata['trial']])
  return metadata


def build_dataset_index(dataset_path, enhanced_path):
  dataset_path = Path(dataset_path)
  enhanced_path = Path(enhanced_path)

  label_paths = {}
  if enhanced_path.exists():
    for label_path in enhanced_path.rglob('*.txt'):
      try:
        stem = normalize_trial_stem(label_path)
        parse_trial_name(stem)
      except ValueError:
        continue
      if stem in label_paths:
        prefix = 'SisfallEnhanced_'
        is_enhanced_name = label_path.stem.startswith(prefix)
        existing_is_enhanced = label_paths[stem].stem.startswith(prefix)
        if is_enhanced_name and not existing_is_enhanced:
          label_paths[stem] = label_path
          continue
        if not is_enhanced_name and existing_is_enhanced:
          continue
        raise ValueError(f'Duplicate enhanced label file for {stem}')
      label_paths[stem] = label_path

  records = []
  if dataset_path.exists():
    for sensor_path in sorted(dataset_path.rglob('*.txt')):
      try:
        metadata = parse_sensor_trial_path(sensor_path)
      except ValueError:
        continue
      metadata.update({
          'sensor_path': sensor_path,
          'label_path': label_paths.get(metadata['trial_stem']),
          'has_enhanced_labels': metadata['trial_stem'] in label_paths,
      })
      records.append(metadata)

  columns = [
      'trial_stem', 'source_trial_stem', 'filename_subject_mismatch',
      'activity', 'activity_type', 'activity_description',
      'subject', 'age_group', 'trial', 'expected_trials',
      'expected_duration_seconds', 'sensor_path', 'label_path',
      'has_enhanced_labels',
  ]
  return pd.DataFrame(records, columns=columns)


def paired_dataset_index(dataset_index):
  """Return only sensor trials that have a matching enhanced-label file.

  The sensor dataset remains the source of recordings. Enhanced-only files are
  never introduced, and sensor recordings without an enhanced counterpart are
  returned separately so callers can report exactly what was skipped.
  """
  if 'has_enhanced_labels' not in dataset_index.columns:
    raise ValueError("dataset_index must contain 'has_enhanced_labels'")

  paired_mask = dataset_index['has_enhanced_labels'].fillna(False)
  paired = dataset_index.loc[paired_mask].copy().reset_index(drop=True)
  skipped = dataset_index.loc[~paired_mask].copy().reset_index(drop=True)
  return paired, skipped


def dataset_overview(dataset_index):
  if dataset_index.empty:
    summary = pd.DataFrame([
        ('Recordings found', 0),
        ('Expected recordings', 4510),
        ('Expected subjects', 38),
        ('Designed ADL activity types', 19),
        ('Designed fall activity types', 15),
        ('Sampling rate', f'{SAMPLING_RATE} Hz'),
        ('Sensor channels', len(SENSOR_COLUMNS)),
    ], columns=['Metric', 'Value'])
    return summary, pd.DataFrame(), pd.DataFrame()

  fall_rows = dataset_index['activity_type'].eq('Fall')
  summary = pd.DataFrame([
      ('Recordings found', len(dataset_index)),
      ('Expected recordings', 4510),
      ('Expected subjects', 38),
      ('Subjects represented', dataset_index['subject'].nunique()),
      ('Adult subjects represented', dataset_index.loc[dataset_index['age_group'].eq('Adult'), 'subject'].nunique()),
      ('Elderly subjects represented', dataset_index.loc[dataset_index['age_group'].eq('Elderly'), 'subject'].nunique()),
      ('ADL activity types', dataset_index.loc[~fall_rows, 'activity'].nunique()),
      ('Fall activity types', dataset_index.loc[fall_rows, 'activity'].nunique()),
      ('ADL recordings', int((~fall_rows).sum())),
      ('Fall recordings', int(fall_rows.sum())),
      ('Fall recordings with enhanced labels', int((fall_rows & dataset_index['has_enhanced_labels']).sum())),
      ('Fall recordings missing enhanced labels', int((fall_rows & ~dataset_index['has_enhanced_labels']).sum())),
      ('Filename/folder subject mismatches corrected', int(
          dataset_index['filename_subject_mismatch'].sum())),
      ('Sampling rate', f'{SAMPLING_RATE} Hz'),
      ('Sensor channels', len(SENSOR_COLUMNS)),
  ], columns=['Metric', 'Value'])

  activity_summary = (dataset_index
      .groupby(['activity', 'activity_type', 'activity_description'], as_index=False)
      .agg(recordings=('trial_stem', 'size'),
           subjects=('subject', 'nunique'),
           enhanced_labels=('has_enhanced_labels', 'sum')))

  subject_summary = (dataset_index
      .groupby(['subject', 'age_group'], as_index=False)
      .agg(recordings=('trial_stem', 'size'),
           activities=('activity', 'nunique'),
           fall_recordings=('activity_type', lambda values: int((values == 'Fall').sum()))))

  return summary, activity_summary, subject_summary


def build_split_manifest(dataset_index, train_paths, test_paths):
  split_by_path = {
      str(Path(path).resolve()): 'train' for path in train_paths
  }
  overlap = set(split_by_path).intersection(
      str(Path(path).resolve()) for path in test_paths)
  if overlap:
    raise ValueError(f'{len(overlap)} recordings occur in both train and test')
  split_by_path.update({
      str(Path(path).resolve()): 'test' for path in test_paths
  })

  manifest = dataset_index.copy()
  manifest['split'] = manifest['sensor_path'].map(
      lambda path: split_by_path.get(str(Path(path).resolve()), 'unassigned'))
  return manifest


def split_overview(split_manifest):
  if split_manifest.empty or 'split' not in split_manifest:
    return pd.DataFrame(), pd.DataFrame()

  train_subjects = set(split_manifest.loc[
      split_manifest['split'].eq('train'), 'subject'])
  test_subjects = set(split_manifest.loc[
      split_manifest['split'].eq('test'), 'subject'])
  summary = pd.DataFrame([
      ('Train recordings', int(split_manifest['split'].eq('train').sum())),
      ('Test recordings', int(split_manifest['split'].eq('test').sum())),
      ('Unassigned recordings', int(split_manifest['split'].eq('unassigned').sum())),
      ('Train subjects', len(train_subjects)),
      ('Test subjects', len(test_subjects)),
      ('Subjects shared by train and test', len(train_subjects & test_subjects)),
      ('Recording overlap', 0),
  ], columns=['Metric', 'Value'])

  activity_split = (split_manifest[split_manifest['split'].isin(['train', 'test'])]
      .groupby(['activity', 'activity_type', 'split'])
      .size()
      .unstack(fill_value=0)
      .reset_index())
  return summary, activity_split


def window_label_overview(window_metadata):
  """Summarize useful class and label-composition information for windows."""
  if window_metadata is None or window_metadata.empty:
    return pd.DataFrame()
  summary = (window_metadata
      .groupby('target', as_index=False)
      .agg(windows=('target', 'size'),
           trials=('trial_stem', 'nunique'),
           mean_no_fall_fraction=('no_fall_fraction', 'mean'),
           mean_pre_fall_fraction=('pre_fall_fraction', 'mean'),
           mean_fall_fraction=('fall_fraction', 'mean')))
  summary.insert(1, 'label', summary['target'].map(LABEL_NAMES))
  return summary.sort_values(
      'target', key=lambda values: values.map({0: 0, 2: 1, 1: 2}))


def read_sensor_trial(path):
  data = pd.read_csv(path, header=None)
  if data.shape[1] != len(SENSOR_COLUMNS):
    raise ValueError(f'{Path(path).name} does not contain nine sensor columns')
  data.columns = SENSOR_COLUMNS
  data['MMA8451Q_z'] = data['MMA8451Q_z'].astype(str).str.rstrip(';')
  return data.apply(pd.to_numeric, errors='raise')


def read_enhanced_labels(path):
  labels = pd.read_csv(path, header=None).iloc[:, 0]
  labels = pd.to_numeric(labels, errors='raise').astype(np.int8)
  invalid = sorted(set(labels.unique()) - set(LABEL_NAMES))
  if invalid:
    raise ValueError(f'{Path(path).name} contains invalid labels: {invalid}')
  return labels


def validate_trial_pair(sensor_path, label_path):
  sensor_data = read_sensor_trial(sensor_path)
  labels = read_enhanced_labels(label_path)
  sensor_stem = normalize_trial_stem(sensor_path)
  label_stem = normalize_trial_stem(label_path)
  if sensor_stem != label_stem:
    raise ValueError(f'Trial mismatch: {sensor_stem} != {label_stem}')
  if len(sensor_data) != len(labels):
    raise ValueError(
        f'Row mismatch for {sensor_stem}: {len(sensor_data)} sensor rows '
        f'and {len(labels)} label rows')
  return sensor_data, labels


def select_fall_trial(dataset_index, preferred_stem=None):
  candidates = dataset_index[
      dataset_index['activity_type'].eq('Fall') &
      dataset_index['has_enhanced_labels']]
  if candidates.empty:
    raise ValueError('No fall trial with enhanced labels was found')
  if preferred_stem:
    preferred = candidates[candidates['trial_stem'].eq(preferred_stem)]
    if not preferred.empty:
      return preferred.iloc[0]
  return candidates.sort_values('trial_stem').iloc[0]


def _label_regions(labels):
  values = labels.to_numpy()
  boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
  starts = np.r_[0, boundaries]
  ends = np.r_[boundaries, len(values)]
  return [(int(start), int(end), int(values[start]))
          for start, end in zip(starts, ends)]


def plot_fall_trial(sensor_path, label_path, window_size=200,
                    convert_units=True, show_window_boundaries=True):
  sensor_data, labels = validate_trial_pair(sensor_path, label_path)
  metadata = parse_trial_name(sensor_path)
  values = sensor_data.to_numpy(dtype=float)

  if convert_units:
    values[:, 0:3] *= (2 * 16) / (2 ** 13)
    values[:, 3:6] *= (2 * 2000) / (2 ** 16)
    values[:, 6:9] *= (2 * 8) / (2 ** 14)

  time_seconds = np.arange(len(values)) / SAMPLING_RATE
  sensors = [
      ('ADXL345', values[:, 0:3], 'Acceleration [g]' if convert_units else 'Raw value [bits]'),
      ('ITG3200', values[:, 3:6], 'Angular velocity [°/s]' if convert_units else 'Raw value [bits]'),
      ('MMA8451Q', values[:, 6:9], 'Acceleration [g]' if convert_units else 'Raw value [bits]'),
  ]
  axes_names = ['X', 'Y', 'Z']

  figure, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
  regions = _label_regions(labels)
  for sensor_index, (sensor_name, sensor_values, unit) in enumerate(sensors):
    for axis_index, axis_name in enumerate(axes_names):
      axis = axes[sensor_index, axis_index]
      for start, end, label in regions:
        axis.axvspan(start / SAMPLING_RATE, end / SAMPLING_RATE,
                     color=LABEL_COLORS[label], alpha=0.22, linewidth=0)
      if show_window_boundaries:
        for boundary in range(window_size, len(values), window_size):
          axis.axvline(boundary / SAMPLING_RATE, color='0.55',
                       linestyle='--', linewidth=0.6, alpha=0.45)
      axis.plot(time_seconds, sensor_values[:, axis_index],
                color='#264653', linewidth=0.8)
      axis.set_title(f'{sensor_name} — {axis_name} axis')
      axis.set_ylabel(unit)
      axis.grid(alpha=0.2)
      if sensor_index == 2:
        axis.set_xlabel('Time [seconds]')

  description = metadata['activity_description']
  figure.suptitle(
      f"{metadata['trial_stem']} — {description}\n"
      'Nine synchronized sensor channels with enhanced fall labels',
      fontsize=14)
  legend = [Patch(facecolor=LABEL_COLORS[label], alpha=0.45,
                  label=f'{label}: {LABEL_NAMES[label]}')
            for label in (0, 2, 1)]
  figure.legend(handles=legend, loc='upper center', ncol=3,
                bbox_to_anchor=(0.5, 0.94))
  figure.tight_layout(rect=(0, 0, 1, 0.89))
  return figure, axes
