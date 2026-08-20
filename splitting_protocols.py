from pathlib import Path

import numpy as np
import pandas as pd


TRIAL_PROTOCOL = 'stratified_trial'
CHRONOLOGICAL_TRIAL_PROTOCOL = 'chronological_trial'
SUBJECT_PROTOCOL = 'subject_independent'
SUPPORTED_PROTOCOLS = {
    TRIAL_PROTOCOL, CHRONOLOGICAL_TRIAL_PROTOCOL, SUBJECT_PROTOCOL}
VALIDATION_PROTOCOL = 'trial_aware_validation'


def _prepare_manifest(dataset_index, protocol, seed):
  if dataset_index is None or dataset_index.empty:
    raise ValueError('A non-empty dataset index is required for splitting')
  required = {'sensor_path', 'trial_stem', 'subject', 'activity', 'activity_type'}
  missing = required - set(dataset_index.columns)
  if missing:
    raise ValueError(f'Dataset index is missing columns: {sorted(missing)}')

  manifest = dataset_index.copy().sort_values(
      ['subject', 'activity', 'trial', 'trial_stem']).reset_index(drop=True)
  manifest['split'] = ''
  manifest['split_protocol'] = protocol
  manifest['split_seed'] = seed
  manifest['assignment_reason'] = ''
  manifest['subject_activity_trial_count'] = manifest.groupby(
      ['subject', 'activity'])['trial_stem'].transform('size')
  return manifest


def stratified_trial_split(dataset_index, train_ratio=0.8, seed=42):
  """Split repetitions within subject/activity groups using a fixed seed.

  Complete five-repetition groups receive exactly four training trials and one
  test trial. Singleton groups are balanced together by activity; incomplete
  multi-trial groups receive at least one trial on each side.
  """
  manifest = _prepare_manifest(dataset_index, TRIAL_PROTOCOL, seed)
  rng = np.random.default_rng(seed)
  singleton_indices = []

  for _, group in manifest.groupby(['subject', 'activity'], sort=True):
    indices = group.index.to_numpy()
    if len(indices) == 1:
      singleton_indices.extend(indices)
      continue

    shuffled = rng.permutation(indices)
    n_test = 1 if len(indices) == 5 else max(1, round(len(indices) * (1 - train_ratio)))
    n_test = min(n_test, len(indices) - 1)
    test_indices = shuffled[:n_test]
    train_indices = shuffled[n_test:]
    manifest.loc[train_indices, ['split', 'assignment_reason']] = [
        'train', 'within-subject/activity repetition split']
    manifest.loc[test_indices, ['split', 'assignment_reason']] = [
        'test', 'within-subject/activity repetition split']

  singleton_rows = manifest.loc[singleton_indices]
  for _, group in singleton_rows.groupby(['activity', 'activity_type'], sort=True):
    indices = group.index.to_numpy()
    shuffled = rng.permutation(indices)
    if len(indices) == 1:
      current_test_fraction = manifest['split'].eq('test').mean()
      n_test = int(current_test_fraction < (1 - train_ratio))
    else:
      n_test = max(1, round(len(indices) * (1 - train_ratio)))
      n_test = min(n_test, len(indices) - 1)
    test_indices = shuffled[:n_test]
    train_indices = shuffled[n_test:]
    manifest.loc[train_indices, ['split', 'assignment_reason']] = [
        'train', 'singleton globally balanced by activity/class']
    manifest.loc[test_indices, ['split', 'assignment_reason']] = [
        'test', 'singleton globally balanced by activity/class']

  validate_split_manifest(manifest)
  five_trial_counts = (manifest[manifest['subject_activity_trial_count'].eq(5)]
      .groupby(['subject', 'activity'])['split'].value_counts().unstack(fill_value=0))
  if not five_trial_counts.empty:
    if not ((five_trial_counts.get('train', 0) == 4) &
            (five_trial_counts.get('test', 0) == 1)).all():
      raise ValueError('The 4-train/1-test rule failed for a five-trial group')
  return manifest


def chronological_trial_split(dataset_index, train_ratio=0.8, seed=42):
  """Split repeated trials chronologically within each subject/activity.

  Earlier repetitions are assigned to training and the latest repetition(s) to
  testing. Activities recorded once per subject cannot be split chronologically,
  so those recordings are reproducibly stratified by activity, activity class,
  and age group. This preserves trial boundaries and keeps both age groups and
  activity classes represented without allowing windows from one trial to leak
  across splits.
  """
  manifest = _prepare_manifest(
      dataset_index, CHRONOLOGICAL_TRIAL_PROTOCOL, seed)
  rng = np.random.default_rng(seed)
  singleton_indices = []

  for _, group in manifest.groupby(['subject', 'activity'], sort=True):
    group = group.sort_values(['trial', 'trial_stem'])
    indices = group.index.to_numpy()
    if len(indices) == 1:
      singleton_indices.extend(indices)
      continue

    n_test = max(1, round(len(indices) * (1 - train_ratio)))
    n_test = min(n_test, len(indices) - 1)
    train_indices = indices[:-n_test]
    test_indices = indices[-n_test:]
    manifest.loc[train_indices, ['split', 'assignment_reason']] = [
        'train', 'earlier within-subject/activity repetition']
    manifest.loc[test_indices, ['split', 'assignment_reason']] = [
        'test', 'latest within-subject/activity repetition']

  singleton_rows = manifest.loc[singleton_indices]
  strata = ['activity', 'activity_type']
  if 'age_group' in singleton_rows.columns:
    strata.append('age_group')
  for _, group in singleton_rows.groupby(strata, sort=True):
    indices = group.sort_values(['subject', 'trial_stem']).index.to_numpy()
    shuffled = rng.permutation(indices)
    if len(indices) == 1:
      current_test_fraction = manifest['split'].eq('test').mean()
      n_test = int(current_test_fraction < (1 - train_ratio))
    else:
      n_test = max(1, round(len(indices) * (1 - train_ratio)))
      n_test = min(n_test, len(indices) - 1)
    test_indices = shuffled[:n_test]
    train_indices = shuffled[n_test:]
    manifest.loc[train_indices, ['split', 'assignment_reason']] = [
        'train', 'singleton stratified by activity/class/age group']
    manifest.loc[test_indices, ['split', 'assignment_reason']] = [
        'test', 'singleton stratified by activity/class/age group']

  validate_split_manifest(manifest)
  five_trial_counts = (manifest[manifest['subject_activity_trial_count'].eq(5)]
      .groupby(['subject', 'activity'])['split'].value_counts().unstack(fill_value=0))
  if not five_trial_counts.empty:
    if not ((five_trial_counts.get('train', 0) == 4) &
            (five_trial_counts.get('test', 0) == 1)).all():
      raise ValueError('The chronological 4-train/1-test rule failed')
  return manifest


def subject_independent_split(dataset_index, train_ratio=0.8, seed=42,
                              candidate_splits=500):
  """Choose a fixed, disjoint subject split with useful activity coverage."""
  manifest = _prepare_manifest(dataset_index, SUBJECT_PROTOCOL, seed)
  subjects = np.array(sorted(manifest['subject'].unique()))
  n_test = min(max(1, round(len(subjects) * (1 - train_ratio))), len(subjects) - 1)
  rng = np.random.default_rng(seed)
  overall_activity = manifest['activity'].value_counts(normalize=True)
  best_subjects = None
  best_score = np.inf

  for _ in range(candidate_splits):
    test_subjects = set(rng.choice(subjects, size=n_test, replace=False))
    test_rows = manifest[manifest['subject'].isin(test_subjects)]
    train_rows = manifest[~manifest['subject'].isin(test_subjects)]
    if (set(test_rows['activity_type']) != set(manifest['activity_type']) or
        set(train_rows['activity_type']) != set(manifest['activity_type'])):
      continue
    test_activity = test_rows['activity'].value_counts(normalize=True)
    activity_error = overall_activity.sub(test_activity, fill_value=0).abs().mean()
    size_error = abs(len(test_rows) / len(manifest) - (1 - train_ratio))
    score = activity_error + size_error
    if score < best_score:
      best_score = score
      best_subjects = test_subjects

  if best_subjects is None:
    raise ValueError('Could not create a subject-independent split with both classes')
  is_test = manifest['subject'].isin(best_subjects)
  manifest.loc[~is_test, ['split', 'assignment_reason']] = [
      'train', 'subject assigned entirely to train']
  manifest.loc[is_test, ['split', 'assignment_reason']] = [
      'test', 'subject assigned entirely to test']
  validate_split_manifest(manifest, require_subject_disjoint=True)
  return manifest


def create_split_manifest(dataset_index, protocol=TRIAL_PROTOCOL,
                          train_ratio=0.8, seed=42):
  if protocol == TRIAL_PROTOCOL:
    return stratified_trial_split(dataset_index, train_ratio=train_ratio, seed=seed)
  if protocol == CHRONOLOGICAL_TRIAL_PROTOCOL:
    return chronological_trial_split(
        dataset_index, train_ratio=train_ratio, seed=seed)
  if protocol == SUBJECT_PROTOCOL:
    return subject_independent_split(dataset_index, train_ratio=train_ratio, seed=seed)
  raise ValueError(
      f'Unsupported split protocol {protocol!r}; choose from {sorted(SUPPORTED_PROTOCOLS)}')


def validate_split_manifest(manifest, require_subject_disjoint=False,
                            require_files=True, expected_protocol=None,
                            expected_seed=None):
  if manifest['trial_stem'].duplicated().any():
    duplicates = manifest.loc[manifest['trial_stem'].duplicated(), 'trial_stem'].tolist()
    raise ValueError(f'Duplicate recordings in split manifest: {duplicates[:5]}')
  invalid_splits = set(manifest['split']) - {'train', 'test'}
  if invalid_splits:
    raise ValueError(f'Unassigned or invalid split values: {sorted(invalid_splits)}')
  if (expected_protocol is not None and
      set(manifest['split_protocol']) != {expected_protocol}):
    raise ValueError('Cached split protocol does not match the active configuration')
  if (expected_seed is not None and
      set(pd.to_numeric(manifest['split_seed'])) != {expected_seed}):
    raise ValueError('Cached split seed does not match the active configuration')
  if (require_files and
      not manifest['sensor_path'].map(lambda path: Path(path).exists()).all()):
    raise FileNotFoundError('At least one split-manifest sensor path does not exist')
  if require_subject_disjoint:
    train_subjects = set(manifest.loc[manifest['split'].eq('train'), 'subject'])
    test_subjects = set(manifest.loc[manifest['split'].eq('test'), 'subject'])
    if train_subjects & test_subjects:
      raise ValueError('Subject-independent split contains shared subjects')
  return True


def paths_from_manifest(manifest):
  train = manifest.loc[manifest['split'].eq('train'), 'sensor_path'].tolist()
  test = manifest.loc[manifest['split'].eq('test'), 'sensor_path'].tolist()
  return train, test


def create_validation_manifest(split_manifest, train_ratio=0.8, seed=43):
  """Create a trial-aware fit/validation split only from training recordings."""
  training_recordings = split_manifest[split_manifest['split'].eq('train')].copy()
  parent_protocol = split_manifest['split_protocol'].iloc[0]
  splitter = (chronological_trial_split
              if parent_protocol == CHRONOLOGICAL_TRIAL_PROTOCOL
              else stratified_trial_split)
  validation_manifest = splitter(
      training_recordings, train_ratio=train_ratio, seed=seed)
  validation_manifest['split'] = validation_manifest['split'].replace({
      'train': 'fit', 'test': 'validation'})
  validation_manifest['split_protocol'] = VALIDATION_PROTOCOL
  validation_manifest['assignment_reason'] = (
      validation_manifest['assignment_reason'] +
      ' (fit/validation within development set)')
  validate_validation_manifest(validation_manifest, expected_seed=seed)
  return validation_manifest


def validate_validation_manifest(validation_manifest, expected_seed=None):
  if validation_manifest['trial_stem'].duplicated().any():
    raise ValueError('Validation manifest contains duplicate recordings')
  if set(validation_manifest['split']) != {'fit', 'validation'}:
    raise ValueError('Fit/validation manifest does not contain both assignments')
  if set(validation_manifest['split_protocol']) != {VALIDATION_PROTOCOL}:
    raise ValueError('Unexpected validation protocol')
  if (expected_seed is not None and
      set(pd.to_numeric(validation_manifest['split_seed'])) != {expected_seed}):
    raise ValueError('Cached validation seed does not match the active configuration')
  return True


def window_indices_from_validation_manifest(window_metadata,
                                            validation_manifest):
  assignment = validation_manifest.set_index('trial_stem')['split']
  window_assignment = window_metadata['trial_stem'].map(assignment)
  if window_assignment.isna().any():
    raise ValueError('At least one training window lacks a validation assignment')
  fit_indices = np.flatnonzero(window_assignment.eq('fit').to_numpy())
  validation_indices = np.flatnonzero(
      window_assignment.eq('validation').to_numpy())
  if len(fit_indices) == 0 or len(validation_indices) == 0:
    raise ValueError('Fit or validation windows are empty')
  return fit_indices, validation_indices


def split_protocol_overview(manifest):
  train_subjects = set(manifest.loc[manifest['split'].eq('train'), 'subject'])
  test_subjects = set(manifest.loc[manifest['split'].eq('test'), 'subject'])
  summary = pd.DataFrame([
      ('Protocol', manifest['split_protocol'].iloc[0]),
      ('Seed', int(manifest['split_seed'].iloc[0])),
      ('Train recordings', int(manifest['split'].eq('train').sum())),
      ('Test recordings', int(manifest['split'].eq('test').sum())),
      ('Train subjects', len(train_subjects)),
      ('Test subjects', len(test_subjects)),
      ('Shared subjects', len(train_subjects & test_subjects)),
      ('Five-trial groups', int(manifest.loc[
          manifest['subject_activity_trial_count'].eq(5),
          ['subject', 'activity']].drop_duplicates().shape[0])),
      ('Singleton recordings', int(manifest['subject_activity_trial_count'].eq(1).sum())),
  ], columns=['Metric', 'Value'])
  singleton_assignments = manifest.loc[
      manifest['subject_activity_trial_count'].eq(1),
      ['trial_stem', 'subject', 'activity', 'activity_type', 'split',
       'assignment_reason']].reset_index(drop=True)
  return summary, singleton_assignments
