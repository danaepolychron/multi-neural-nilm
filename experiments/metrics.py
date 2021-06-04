import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix, roc_curve

METRICS = {"Accuracy" : accuracy_score, 
           "Precision" : precision_score, 
           "Recall" : recall_score, 
           "F1 Score" : f1_score, 
           "MCC" : matthews_corrcoef}



mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def multilabelify(metric, y_true, y_predicted, mode="target"):
    
    """Returns per label or averaged results of a scikit-learn compatible quality metric.

    Parameters
    ----------
    metric : callable scikit-compatible quality metric function
    y_true : numpy.array, ground truth
    y_predicted : numpy.array, the predicted result
    mode: str, either "target" or "averaged"

    Returns
    -------
    List[int or float], scores from a given metric

    """
    
    if mode == "target":
        return [ metric(y_true[:, i], y_predicted[:, i]) for i in range(y_true.shape[1]) ]
    
    elif mode == "averaged":
        return [ metric(y_true, y_predicted, average=i) for i in ["macro", "micro"] ]
    else:
        return None
    
    
def validation_report(history):
    
    """Creates a report on training and validation losses and metrics at successive epochs.

    Parameters
    ----------
    history : tf.keras.callback.History, its History.history attribute is a record of training 
              loss values and metrics values at successive epochs, as well as validation loss values 
              and validation metrics values (if applicable).

    """
    
    
    metrics = [k for k in history.history.keys() if "val" not in k]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20,5))
    
    for i, metric in enumerate(metrics):
        
        name = metric.replace("_"," ").capitalize()
        
        axes[i].plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        axes[i].plot(history.epoch, history.history['val_'+ metric], color=colors[0], linestyle="--", label='Validation')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(name)
        axes[i].legend()
        
        if metric == 'loss':
            axes[i].set_ylim([0, axes[i].get_ylim()[1]])
        else:
            axes[i].set_ylim([0,1])
  
    
def evaluation_report(y_true, y_predicted, targets=None, metrics=METRICS, mode="target"):
    
    """Creates a report on a set of metrics to evaluate the classifier's performance. 

    Parameters
    ----------
    y_true : numpy.array, ground truth
    y_predicted : numpy.array, the predicted result
    mode: str, either "target" or "averaged"
    
    Returns
    -------
    pandas.DataFrame, a dataframe with the specified metric scores

    """
    
    if mode == "target": 
        
        results = {}
    
        for i, metric in metrics.items() :

            results[i]= multilabelify(metric, y_true, y_predicted, mode)
        
        return pd.DataFrame(results, columns=metrics.keys(), index=targets)
    
    elif mode == "averaged":
        
        keys = ['Precision', 'Recall','F1 Score']
        
        avg_metrics = {key: metrics[key] for key in keys}

        results = {}
        for i, metric in avg_metrics.items() :
            
            results[i] = multilabelify(metric, y_true, y_predicted, mode)
            
        return pd.DataFrame(results, columns=keys, index=["macro", "micro"])
    
    else:
        return None


    
def confusion_matrix_report(y_true, y_predicted, targets):  
    
    """Creates the corresponding confusion matrices for all targets. 

    Parameters
    ----------
    y_true : numpy.array, ground truth
    y_predicted : numpy.array, the predicted result
    targets: str, target name

    """
    
    results = multilabelify(confusion_matrix, y_true, y_predicted, mode="target")
    
    size = len(targets)
    fig, axes = plt.subplots(size, figsize=(20,20))
    fig.suptitle('Confusion Matrix for each appliance')
    
    for i, (target, cm) in enumerate(zip(targets, results)): 
    
        axes[i] = sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title('Confusion Matrix for {}'.format(target))
        axes[i].set_ylabel('Actual label')
        axes[i].set_xlabel('Predicted label')
        axes[i].set_aspect('equal')
        

def roc_report(y_true, y_predicted, targets): 
    
    """Creates the corresponding roc curves for all targets. 

    Parameters
    ----------
    y_true : numpy.array, ground truth
    y_predicted : numpy.array, the predicted result
    targets: str, target name

    """
    
    results = multilabelify(roc_curve, y_true, y_predicted, mode="target")
    
    size = len(targets)
    fig, axes = plt.subplots(size, figsize=(20,20))
    fig.suptitle('ROC for each appliance')
    
    for i, (target, result) in enumerate(zip(targets,results)): 
        
        fp, tp, _ = result
        axes[i].plot(100*fp, 100*tp,linewidth=2)
        axes[i].set_title('ROC for {}'.format(target))
        axes[i].set_ylabel('False positives [%]')
        axes[i].set_xlabel('True positives [%]')

        axes[i].grid(True)
        axes[i].set_aspect('equal')
        
