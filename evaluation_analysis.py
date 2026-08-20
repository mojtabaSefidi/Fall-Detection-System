from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (auc, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support)

from utils import CLASS_LABELS, CLASS_NAMES


def _validate_inputs(y_true, predictions, metadata=None):
  y_true = np.asarray(y_true).ravel()
  for model_name, prediction in predictions.items():
    if len(prediction) != len(y_true):
      raise ValueError(
          f'{model_name} has {len(prediction)} predictions for {len(y_true)} labels')
  if metadata is not None and len(metadata) != len(y_true):
    raise ValueError('Window metadata and test labels are not aligned')
  return y_true


def _class_scores(model_name, probabilities, performance, class_label):
  if model_name not in probabilities:
    return None
  scores = np.asarray(probabilities[model_name])
  if scores.ndim != 2:
    return None
  class_order = performance.get(model_name, {}).get(
      'probability_classes', list(range(scores.shape[1])))
  if class_label not in class_order:
    return None
  return scores[:, class_order.index(class_label)]


def focused_model_report(y_true, predictions, probabilities=None,
                         performance=None, window_metadata=None):
  """Build the compact result table used as the main experiment report."""
  probabilities = probabilities or {}
  performance = performance or {}
  y_true = _validate_inputs(y_true, predictions, window_metadata)
  if window_metadata is not None:
    adl_mask = window_metadata.reset_index(drop=True)['activity'].str.startswith('D').to_numpy()
    if {'start_time_seconds', 'end_time_seconds'} <= set(window_metadata):
      durations = (window_metadata['end_time_seconds'] -
                   window_metadata['start_time_seconds']).to_numpy()
      adl_hours = durations[adl_mask].sum() / 3600
    else:
      adl_hours = np.count_nonzero(adl_mask) / 3600
  else:
    adl_mask = y_true == 0
    adl_hours = np.count_nonzero(adl_mask) / 3600
  rows = []

  for model_name, prediction in predictions.items():
    prediction = np.asarray(prediction).ravel()
    fall_precision, fall_recall, fall_f1, _ = precision_recall_fscore_support(
        y_true, prediction, labels=[1], average=None, zero_division=0)
    fall_scores = _class_scores(
        model_name, probabilities, performance, class_label=1)
    if fall_scores is not None:
      curve_precision, curve_recall, _ = precision_recall_curve(
          y_true == 1, fall_scores)
      pr_auc = auc(curve_recall, curve_precision)
    else:
      pr_auc = np.nan
    false_negatives = int(np.count_nonzero((y_true == 1) & (prediction != 1)))
    false_positives = int(np.count_nonzero((y_true != 1) & (prediction == 1)))
    adl_false_positives = int(np.count_nonzero(adl_mask & (prediction == 1)))
    model_performance = performance.get(model_name, {})
    rows.append({
        'model': model_name,
        'fall_precision': fall_precision[0],
        'fall_recall': fall_recall[0],
        'fall_f1': fall_f1[0],
        'fall_pr_auc': pr_auc,
        'balanced_accuracy': balanced_accuracy_score(y_true, prediction),
        'macro_f1': f1_score(
            y_true, prediction, labels=CLASS_LABELS, average='macro',
            zero_division=0),
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'false_alarms_per_adl_hour': (adl_false_positives / adl_hours
                                      if adl_hours else np.nan),
        'training_seconds': model_performance.get('training_seconds', np.nan),
        'inference_ms_per_window': model_performance.get(
            'inference_ms_per_window', np.nan),
        'model_size_mb': model_performance.get('model_size_bytes', np.nan) / (1024 ** 2),
    })
  return pd.DataFrame(rows).sort_values(
      ['macro_f1', 'fall_recall'], ascending=False).reset_index(drop=True)


def activity_error_reports(y_true, predictions, window_metadata):
  """Return per-activity false positives and per-fall-type false negatives."""
  y_true = _validate_inputs(y_true, predictions, window_metadata)
  base = window_metadata.reset_index(drop=True)[
      ['activity', 'trial_stem']].copy()
  reports = []
  for model_name, prediction in predictions.items():
    frame = base.copy()
    frame['actual'] = y_true
    frame['predicted'] = np.asarray(prediction).ravel()
    grouped = (frame.groupby('activity', as_index=False)
        .agg(windows=('actual', 'size'),
             actual_falls=('actual', lambda values: int((values == 1).sum()))))
    false_positive_counts = frame[(frame.actual == 0) & (frame.predicted == 1)][
        'activity'].value_counts()
    false_negative_counts = frame[(frame.actual == 1) & (frame.predicted != 1)][
        'activity'].value_counts()
    grouped['fall_false_positives'] = grouped['activity'].map(
        false_positive_counts).fillna(0).astype(int)
    grouped['fall_false_negatives'] = grouped['activity'].map(
        false_negative_counts).fillna(0).astype(int)
    grouped['model'] = model_name
    reports.append(grouped)
  report = pd.concat(reports, ignore_index=True)
  false_positives = report[report['fall_false_positives'].gt(0)].sort_values(
      ['model', 'fall_false_positives'], ascending=[True, False])
  false_negatives = report[
      report['activity'].str.startswith('F') & report['fall_false_negatives'].gt(0)
  ].sort_values(['model', 'fall_false_negatives'], ascending=[True, False])
  return false_positives.reset_index(drop=True), false_negatives.reset_index(drop=True)


def subject_performance_report(y_true, predictions, window_metadata):
  y_true = _validate_inputs(y_true, predictions, window_metadata)
  reports = []
  subjects = window_metadata.reset_index(drop=True)['subject'].to_numpy()
  age_groups = window_metadata.reset_index(drop=True)['age_group'].to_numpy()
  for model_name, prediction in predictions.items():
    prediction = np.asarray(prediction).ravel()
    for subject in np.unique(subjects):
      selected = subjects == subject
      fall_windows = int(np.count_nonzero(y_true[selected] == 1))
      fall_recall = (precision_recall_fscore_support(
          y_true[selected], prediction[selected], labels=[1], average=None,
          zero_division=0)[1][0] if fall_windows else np.nan)
      reports.append({
          'model': model_name,
          'subject': subject,
          'age_group': age_groups[selected][0],
          'windows': int(selected.sum()),
          'fall_windows': fall_windows,
          'macro_f1': f1_score(
              y_true[selected], prediction[selected], labels=CLASS_LABELS,
              average='macro', zero_division=0),
          'fall_recall': fall_recall,
      })
  return pd.DataFrame(reports).sort_values(['model', 'macro_f1'])


def window_composition_report(window_metadata):
  return (window_metadata.groupby('target', as_index=False)
      .agg(windows=('target', 'size'),
           mean_pre_fall_fraction=('pre_fall_fraction', 'mean'),
           median_pre_fall_fraction=('pre_fall_fraction', 'median'),
           mean_fall_fraction=('fall_fraction', 'mean'),
           median_fall_fraction=('fall_fraction', 'median'),
           median_pre_fall_start=('pre_fall_start_fraction', 'median'),
           median_fall_start=('fall_start_fraction', 'median'))
      .round(4))


def bootstrap_confidence_intervals(y_true, predictions, window_metadata,
                                   iterations=200, seed=42, confidence=0.95):
  """Cluster-bootstrap metrics by trial so correlated windows stay together."""
  y_true = _validate_inputs(y_true, predictions, window_metadata)
  if iterations <= 0:
    return pd.DataFrame()
  trial_values = window_metadata.reset_index(drop=True)['trial_stem'].to_numpy()
  trials = np.unique(trial_values)
  indices_by_trial = {trial: np.flatnonzero(trial_values == trial) for trial in trials}
  rng = np.random.default_rng(seed)
  values = {model_name: {'macro_f1': [], 'balanced_accuracy': [], 'fall_recall': []}
            for model_name in predictions}

  for _ in range(iterations):
    sampled_trials = rng.choice(trials, size=len(trials), replace=True)
    selected = np.concatenate([indices_by_trial[trial] for trial in sampled_trials])
    actual = y_true[selected]
    for model_name, prediction in predictions.items():
      predicted = np.asarray(prediction).ravel()[selected]
      values[model_name]['macro_f1'].append(f1_score(
          actual, predicted, labels=CLASS_LABELS, average='macro', zero_division=0))
      values[model_name]['balanced_accuracy'].append(
          balanced_accuracy_score(actual, predicted))
      values[model_name]['fall_recall'].append(
          precision_recall_fscore_support(
              actual, predicted, labels=[1], average=None,
              zero_division=0)[1][0])

  alpha = (1 - confidence) / 2
  rows = []
  for model_name, metrics in values.items():
    prediction = np.asarray(predictions[model_name]).ravel()
    original_estimates = {
        'macro_f1': f1_score(
            y_true, prediction, labels=CLASS_LABELS, average='macro',
            zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, prediction),
        'fall_recall': precision_recall_fscore_support(
            y_true, prediction, labels=[1], average=None,
            zero_division=0)[1][0],
    }
    for metric, samples in metrics.items():
      rows.append({
          'model': model_name,
          'metric': metric,
          'estimate': float(original_estimates[metric]),
          'ci_lower': float(np.quantile(samples, alpha)),
          'ci_upper': float(np.quantile(samples, 1 - alpha)),
          'bootstrap_iterations': iterations,
      })
  return pd.DataFrame(rows)


def plot_fall_precision_recall(y_true, probabilities, performance=None):
  performance = performance or {}
  figure, axis = plt.subplots(figsize=(9, 6))
  for model_name in probabilities:
    scores = _class_scores(model_name, probabilities, performance, class_label=1)
    if scores is None:
      continue
    precision, recall, _ = precision_recall_curve(np.asarray(y_true) == 1, scores)
    pr_auc = auc(recall, precision)
    axis.plot(recall, precision, label=f'{model_name} ({pr_auc:.3f})')
  axis.set(title='Fall precision–recall comparison', xlabel='Recall',
           ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
  axis.legend(title='Model (PR-AUC)', bbox_to_anchor=(1.02, 1), loc='upper left')
  axis.grid(alpha=0.2)
  figure.tight_layout()
  return figure, axis


def plot_metric_comparison(focused_report):
  metrics = ['fall_precision', 'fall_recall', 'fall_f1',
             'balanced_accuracy', 'macro_f1']
  plot_data = focused_report.set_index('model')[metrics]
  axis = plot_data.plot.bar(figsize=(12, 6), ylim=(0, 1), rot=25)
  axis.set(title='Focused model comparison', ylabel='Score', xlabel='')
  axis.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
  axis.grid(axis='y', alpha=0.2)
  axis.figure.tight_layout()
  return axis.figure, axis


def plot_confusion_matrix_grid(y_true, predictions):
  n_models = len(predictions)
  n_columns = 2
  n_rows = int(np.ceil(n_models / n_columns))
  figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4.5 * n_rows))
  axes = np.atleast_1d(axes).ravel()
  for axis, (model_name, prediction) in zip(axes, predictions.items()):
    matrix = confusion_matrix(y_true, prediction, labels=CLASS_LABELS)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0)
    annotations = np.asarray([
        f'{count}\n{percentage:.1%}'
        for count, percentage in zip(matrix.ravel(), percentages.ravel())
    ]).reshape(matrix.shape)
    sns.heatmap(matrix, annot=annotations, fmt='', cmap='Greens', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(title=model_name, xlabel='Predicted label', ylabel='Actual label')
  for axis in axes[n_models:]:
    axis.set_visible(False)
  figure.suptitle('Confusion matrices: counts and row percentages')
  figure.tight_layout()
  return figure, axes


def save_analysis_tables(output_path, **tables):
  output_path = Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  for name, table in tables.items():
    if table is not None:
      table.to_csv(output_path / f'{name}.csv', index=False)


def collect_experiment_reports(result_root):
  """Combine completed protocol/mode reports without mixing their identities."""
  result_root = Path(result_root)
  reports = []
  for report_path in sorted(result_root.glob(
      '*/*/window-*/seed-*/analysis/focused_model_report.csv')):
    relative = report_path.relative_to(result_root)
    protocol, preprocessing_mode, window_directory, seed_directory = relative.parts[:4]
    report = pd.read_csv(report_path)
    report.insert(0, 'random_seed', int(seed_directory.split('-', 1)[1]))
    report.insert(0, 'window_size', int(window_directory.split('-', 1)[1]))
    report.insert(0, 'preprocessing_mode', preprocessing_mode)
    report.insert(0, 'split_protocol', protocol)
    reports.append(report)
  return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
