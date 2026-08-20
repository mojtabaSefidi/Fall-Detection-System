"""Focused binary evaluation for the standard baseline experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


CLASS_LABELS = [0, 1]
CLASS_NAMES = ['No event', 'Pre-fall or fall']


def _safe_auc(metric, y_true, scores):
  return metric(y_true, scores) if np.unique(y_true).size == 2 else np.nan


def binary_summary(y_true, predictions, probabilities, performance=None):
  """Create one standard comparison row per model."""
  performance = performance or {}
  rows = []
  for model_name, prediction in predictions.items():
    prediction = np.asarray(prediction).reshape(-1)
    scores = np.asarray(probabilities[model_name]).reshape(-1)
    timing = performance.get(model_name, {})
    rows.append({
        'model': model_name,
        'accuracy': accuracy_score(y_true, prediction),
        'balanced_accuracy': balanced_accuracy_score(y_true, prediction),
        'precision': precision_score(y_true, prediction, zero_division=0),
        'recall': recall_score(y_true, prediction, zero_division=0),
        'f1': f1_score(y_true, prediction, zero_division=0),
        'roc_auc': _safe_auc(roc_auc_score, y_true, scores),
        'pr_auc': _safe_auc(average_precision_score, y_true, scores),
        'training_seconds': timing.get('training_seconds', np.nan),
        'inference_ms_per_window': timing.get(
            'inference_ms_per_window', np.nan),
    })
  return pd.DataFrame(rows).sort_values(
      ['f1', 'pr_auc'], ascending=False).reset_index(drop=True)


def classification_report_tables(y_true, predictions):
  reports = {}
  for model_name, prediction in predictions.items():
    report = classification_report(
        y_true, prediction, labels=CLASS_LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    reports[model_name] = pd.DataFrame(report).T
  return reports


def plot_confusion_matrices(y_true, predictions):
  n_models = len(predictions)
  n_columns = 2
  n_rows = int(np.ceil(n_models / n_columns))
  figure, axes = plt.subplots(
      n_rows, n_columns, figsize=(11, 4.2 * n_rows), squeeze=False)
  axes = axes.ravel()
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
    sns.heatmap(
        matrix, annot=annotations, fmt='', cmap='Blues', cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(title=model_name, xlabel='Predicted', ylabel='Actual')
  for axis in axes[n_models:]:
    axis.set_visible(False)
  figure.suptitle('Confusion matrices (count and row percentage)', y=1.01)
  figure.tight_layout()
  return figure


def plot_binary_curves(y_true, probabilities):
  figure, axes = plt.subplots(1, 2, figsize=(13, 5))
  for model_name, scores in probabilities.items():
    scores = np.asarray(scores).reshape(-1)
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    axes[0].plot(
        fpr, tpr,
        label=f'{model_name} ({roc_auc_score(y_true, scores):.3f})')
    axes[1].plot(
        recall, precision,
        label=f'{model_name} ({average_precision_score(y_true, scores):.3f})')
  axes[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
  axes[0].set(
      title='ROC curves', xlabel='False-positive rate',
      ylabel='True-positive rate', xlim=(0, 1), ylim=(0, 1.02))
  axes[1].axhline(
      np.mean(y_true), linestyle='--', color='grey', linewidth=1,
      label=f'Positive prevalence ({np.mean(y_true):.3f})')
  axes[1].set(
      title='Precision–recall curves', xlabel='Recall', ylabel='Precision',
      xlim=(0, 1), ylim=(0, 1.02))
  for axis in axes:
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
  figure.tight_layout()
  return figure


def plot_cnn_history(history):
  figure, axes = plt.subplots(1, 2, figsize=(12, 4))
  axes[0].plot(history.history['loss'], label='Train')
  axes[0].plot(history.history['val_loss'], label='Validation')
  axes[0].set(title='CNN loss', xlabel='Epoch', ylabel='Binary cross-entropy')
  axes[1].plot(history.history['pr_auc'], label='Train')
  axes[1].plot(history.history['val_pr_auc'], label='Validation')
  axes[1].set(title='CNN PR-AUC', xlabel='Epoch', ylabel='PR-AUC', ylim=(0, 1))
  for axis in axes:
    axis.grid(alpha=0.2)
    axis.legend()
  figure.tight_layout()
  return figure


def save_binary_reports(output_path, summary, reports):
  output_path = Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)
  summary.to_csv(output_path / 'model_summary.csv', index=False)
  for model_name, report in reports.items():
    safe_name = model_name.lower().replace(' ', '_')
    report.to_csv(output_path / f'{safe_name}_classification_report.csv')
