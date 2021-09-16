import numpy as np
from sklearn import metrics
import torch

from models.sequential import Sequential
from models.multitask import Multitask
from models.replay import Replay
from models.oml import OML
from models.anml import ANML
from models.drill import DRILL
from models.soinn_models import SOINNPLUS


def init_model(device, n_classes, **kwargs):
    learner = kwargs.get('learner').upper()

    # Get the model
    if learner == 'SEQUENTIAL':
        return Sequential(device, n_classes, **kwargs)
    elif learner == 'MULTITASK':
        return Multitask(device, n_classes, **kwargs)
    elif learner == 'REPLAY':
        return Replay(device, n_classes, **kwargs)
    elif learner == 'OML':
        return OML(device, n_classes, **kwargs)
    elif learner == 'ANML':
        return ANML(device, n_classes, fusion="gate", **kwargs)
    elif learner == 'DRILLMUL':
        return DRILL(device, n_classes, SOINNPLUS(**kwargs), fusion="gate", **kwargs)
    elif learner == 'DRILLCONCAT' or learner == 'DRILL':
        return DRILL(device, n_classes, SOINNPLUS(**kwargs), fusion="cat", **kwargs)
    else:
        raise Exception('The requested learning model {} is not valid. Check typos or implementation.'.format(learner))


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def calculate_metrics(predictions, labels):
    """
    Calculate accuracy, precision, recall, and F1 score
    :param predictions: model output for given batch samples
    :param labels: true class labels
    :return: metric scores
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='macro', labels=unique_labels, zero_division=0)
    recall = metrics.recall_score(labels, predictions, average='macro', labels=unique_labels, zero_division=0)
    f1_score = metrics.f1_score(labels, predictions, average='macro', labels=unique_labels, zero_division=0)
    return accuracy, precision, recall, f1_score
