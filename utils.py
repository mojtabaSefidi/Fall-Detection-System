from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sn
import pandas as pd
import numpy as np
import pickle
sn.set()



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
CLASS_LABELS = [0, 2, 1]
CLASS_NAMES = ['no fall', 'pre-fall', 'fall']

def plot_confusion_matrix(confusion_matrix, title='', cmap='Greens', class_names=None):
    class_names = class_names or CLASS_NAMES[:len(confusion_matrix)]
    df = pd.DataFrame(confusion_matrix, class_names, class_names)
    plt.figure(figsize=(7,4))
    if title == '' :
        plt.title('Confusion Matrix')
    else:
        plt.title('Confusion Matrix for' + ' ' + title)
    sn.set(font_scale=1) # for label size
    sn.heatmap(df, annot=True, annot_kws={"size": 12},fmt='.0f',cmap=cmap) # font size
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_precision_recall_curve(actual_labels, prediction, title='', model_name='', file_name=None):
    fig, ax = plt.subplots(figsize=(8,6))
    if np.ndim(prediction) == 1 or np.shape(prediction)[-1] == 1:
        precision, recall, _ = precision_recall_curve(actual_labels, np.ravel(prediction))
        ax.plot(recall, precision, label=f'Model: {model_name}', color='purple')
    else:
        encoded = label_binarize(actual_labels, classes=np.arange(prediction.shape[1]))
        for class_id, class_name in zip(CLASS_LABELS, CLASS_NAMES):
            precision, recall, _ = precision_recall_curve(encoded[:, class_id], prediction[:, class_id])
            ax.plot(recall, precision, label=class_name)

    # add axis labels to plot
    plt.title(title or 'Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend()

    # display plot
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()

def plot_roc_curve(actual_labels, prediction, title='', model_name='', file_name=None):
    plt.figure(figsize=(8,6))
    if np.ndim(prediction) == 1 or np.shape(prediction)[-1] == 1:
        fpr, tpr, _ = roc_curve(actual_labels, np.ravel(prediction))
        plt.plot(fpr, tpr, label=f'Model: {model_name}', color='blue')
    else:
        encoded = label_binarize(actual_labels, classes=np.arange(prediction.shape[1]))
        for class_id, class_name in zip(CLASS_LABELS, CLASS_NAMES):
            fpr, tpr, _ = roc_curve(encoded[:, class_id], prediction[:, class_id])
            plt.plot(fpr, tpr, label=class_name)
    plt.title(title or 'ROC Learning Curves')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend()
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()

def plot_metrics(history):
    metrics = [key for key in ('loss', 'Accuracy') if key in history.history]
    plt.figure(figsize=(10, 4 * len(metrics)), linewidth=7, edgecolor="whitesmoke")

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(len(metrics), 1, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric.lower() == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

def plot_auc_curve(actual_labels, prediction, title='', model_name='', file_name = None):
    plt.figure(figsize=(8,6))
    plt.title(title or 'AUC Learning Curves')
    if np.ndim(prediction) == 1 or np.shape(prediction)[-1] == 1:
        prediction = np.ravel(prediction)
        fpr, tpr, _ = roc_curve(actual_labels, prediction)
        auc = roc_auc_score(actual_labels, prediction).round(4)
        plt.plot(fpr, tpr, label=f'Model: {model_name}, AUC={auc}', color='red')
    else:
        encoded = label_binarize(actual_labels, classes=np.arange(prediction.shape[1]))
        for class_id, class_name in zip(CLASS_LABELS, CLASS_NAMES):
            fpr, tpr, _ = roc_curve(encoded[:, class_id], prediction[:, class_id])
            auc = roc_auc_score(encoded[:, class_id], prediction[:, class_id]).round(4)
            plt.plot(fpr, tpr, label=f'{class_name}, AUC={auc}')
    plt.legend(loc=4)
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()

def plot_history(history):

    plt.figure(figsize=(10,5),linewidth = 7, edgecolor="whitesmoke")
    n = len(history.history['Accuracy'])

    plt.plot(np.arange(0,n)+1,history.history['Accuracy'], color='orange',marker=".")
    plt.plot(np.arange(0,n)+1,history.history['loss'],'b',marker=".")

    # offset both validation curves
    plt.plot(np.arange(0,n)+ 1,history.history['val_Accuracy'],'r')
    plt.plot(np.arange(0,n)+ 1,history.history['val_loss'],'g')

    plt.legend(['Train Acc','Train Loss','Val Acc','Val Loss'])
    plt.grid(True)

    # set vertical limit to 1
    plt.gca().set_ylim(0,1)

    plt.xlabel("Number of Epochs")
    plt.ylabel("Value")
    plt.suptitle("Learning Curve", size=16, y=0.927)
    plt.show()

def save_pickle(variable, path):
  with open(path,'wb') as f:
    pickle.dump(variable, f)
    return

def load_pickle(path):
  with open(path,'rb') as f:
    return pickle.load(f)

def get_classifiaction_metrics():
  return get_classification_metrics()

def get_classification_metrics():
  return [keras.metrics.SparseCategoricalAccuracy(name='Accuracy')]

def get_early_stopping(metric='val_loss', patience=5, mode='min'):
  early_stopping = keras.callbacks.EarlyStopping(monitor=metric,
                                                verbose=1,
                                                patience=patience,
                                                mode=mode,
                                                restore_best_weights=True)
  return early_stopping

def convert_report_df(report):
  d = {}
  for key, value in report.items():
      temp = {}
      for label, metrics in value.items():
        if type(metrics)==dict:
          for metric, score in metrics.items():
            temp[metric+'_'+label] = score
        else:
          temp[label] = metrics
      d[key] = temp
  return pd.DataFrame.from_dict(d).T.style.background_gradient(cmap="PuBu")
