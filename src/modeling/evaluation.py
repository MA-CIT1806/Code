import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sn
from datetime import datetime
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 7), title=""):
    """Given true labels and predicted labels, create a confusion matrix. Used during tensorboard-logging."""
    
    class_names = list(class_names)
    available_classes = np.unique(np.concatenate([y_true, y_pred], axis=None))
    available_class_names = [class_names[i] for i in available_classes]
    
    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm, index=available_class_names,
                         columns=available_class_names)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    sn.heatmap(df_cm, annot=True)
    return fig


def create_temporal_softmax_data(y_true, y_pred_raw, mapping):
    """Creates a visualization of the development of log-probabilities over time. Used during tensorboard-logging."""

    my_dict = {}
    for i, label in enumerate(mapping.values()):
        indices = y_true == i
        temp_pred_raw = y_pred_raw[indices]
        
        my_list = []
        
        for j in range(temp_pred_raw.shape[0]):
            temp_dict = {}
            for k in range(temp_pred_raw.shape[1]):
                temp_dict["class_{}".format(mapping[k])] = temp_pred_raw[j, k]
                
            my_list.append(temp_dict)    
            
        my_dict[label] = my_list
               
    return my_dict  

def plot_confusion_matrix(*args, **kwargs):
    _ = create_confusion_matrix(*args, **kwargs)
    plt.show()


def plot_metric_lines(x, y, figsize=(16, 8), title="placeholder", xlabel="Epochs", ylabel=""):
    """Plot multiple metrics in one graphic."""
    plt.figure(figsize=figsize)
    plt.title(title)
    for key, value in y.items():
        plt.plot(x, value, label=key)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
