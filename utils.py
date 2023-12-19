from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sn
import pandas as pd
import pickle
sn.set()



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_confusion_matrix(confusion_matrix, title='', cmap ='Greens'):
    df = pd.DataFrame(confusion_matrix, ['not fall', 'fall'], ['not fall', 'fall'])
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
    precision, recall, thresholds=precision_recall_curve(actual_labels, prediction)
    plt.figure(figsize=(8,6))
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Model: {model_name}'.format(model_name=model_name), color='purple')

    # add axis labels to plot
    if title == '':
        plt.title(title)
    else:
        plt.title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)

def plot_roc_curve(actual_labels, prediction, title='', model_name='', file_name=None):
    fpr, tpr, _ = roc_curve(actual_labels, prediction)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='Model: {model_name}'.format(model_name=model_name), color='blue')
    if title == '':
        plt.title(title)
    else:
        plt.title('ROC Learning Curves')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()

    if file_name is not None:
        plt.savefig(file_name)

def plot_metrics(history):
    metrics = ['loss', 'PRC', 'Precision', 'Recall']
    plt.figure(figsize=(10,10),linewidth = 7, edgecolor="whitesmoke")

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
    if metric == 'loss':
        plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
        plt.ylim([0.8,1])
    else:
        plt.ylim([0,1])

    plt.legend()

def plot_auc_curve(actual_labels, prediction, title='', model_name='', file_name = None):
    fpr, tpr, _ = roc_curve(actual_labels,  prediction)
    auc = roc_auc_score(actual_labels, prediction).round(4)
    plt.figure(figsize=(8,6))
    if title == '':
        plt.title(title)
    else:
        plt.title('AUC Learning Curves')
    plt.plot(fpr,tpr, label='Model: '+ model_name + ", AUC=" + str(auc), color='red')
    plt.legend(loc=4)
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)

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
  metrics = [keras.metrics.TruePositives(name='TP'),
             keras.metrics.FalsePositives(name='FP'),
             keras.metrics.TrueNegatives(name='TN'),
             keras.metrics.FalseNegatives(name='FN'),
             keras.metrics.BinaryAccuracy(name='Accuracy'),
             keras.metrics.Precision(name='Precision'),
             keras.metrics.Recall(name='Recall'),
             keras.metrics.AUC(name='AUC'),
             keras.metrics.AUC(name='PRC', curve='PR')]
  return metrics

def get_early_stopping(metric='val_PRC', patience=5, mode='max'):
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
