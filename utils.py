# Useful Metric class implementations for PyTorch mimicking tf.keras.metric API
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score, \
        precision_recall_curve, auc

class Accuracy:
    """ Update Accuracy in online """
    def __init__(self):
        self._num_samples = 0
        self._num_corrects = 0

    def reset_state(self):
        """ Reset internal state of a Metric class"""
        self.__init__()

    def update_state(self, y_pred, y_true):
        """ Update internal state of Metric class
        Args:
            y_pred (torch.Tensor): class probability or logits.
                2-d tensor of size [num_samples, num_classes].
            y_true (torch.Tensor): groudtruth class labels encoded with
                integer in [0, num_classes-1]. 1d-tensor of size [num_samples].
        Returns:
            None
        """
        self._num_samples += y_pred.size(0)
        y_pred_int = torch.argmax(y_pred, 1)
        self._num_corrects += torch.sum(y_pred_int == y_true.data)

    def result(self):
        """ Compute metric and return it """
        return float(self._num_corrects) / self._num_samples

class Mean:
    def __init__(self):
        self._num_samples = 0
        self._sum = 0.0

    def reset_state(self):
        self.__init__()

    def update_state(self, inputs):
        self._num_samples += inputs.size(0)
        self._sum = torch.sum(inputs)

    def result(self):
        return self._sum/self._num_samples

#class AUC:
#    """ Metric call for computing AUC """
#    def __init__(self,
#                 num_thresholds=200,
#                 curve='ROC',
#                 summation_method='interpolation',
#                 thresholds=None,
#                 multi_label=False,
#                 label_weights=None):
#        """ initialize AUC instance for details see TensorFlow doc """
#        self.num_thresholds = num_thresholds
#        self.curve = curve
#        self.summation_method = summation_method
#        self.thresholds = thresholds
#        self.multi_label = multi_label
#        self.label_weights = label_weights
#
#        self.reset_state()
#
#    def reset_state(self):
#        self.tp = 0 # True positive
#        self.tn = 0 # True negative
#        self.fp = 0 # False positive
#        self.fn = 0 # False negative
#
#    def update_state(self, y_pred, y_true, sample_weights):
#        if thresholds is None:
#            self.num_thresholds
#
#    def result(self):
#        pass

class SimpleAUC:
    """ Simple and accurate but consume much more memory than AUC class """
    def __init__(self, curve='ROC'):
        if curve not in ['ROC', 'PR']:
            raise ValueError('Invalid argument for curve: {}'.format(curve))
        self.curve = curve
        self.y_pred = None
        self.y_true = None

    def reset_state(self):
        self.__init__()

    def numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)

    def update_state(self, y_pred, y_true):
        if self.y_pred is None:
            self.y_pred = self.numpy(y_pred)
        else:
            self.y_pred = np.concatenate((self.y_pred, self.numpy(y_pred)), 0)

        if self.y_true is None:
            self.y_true = self.numpy(y_true)
        else:
            self.y_true = np.concatenate((self.y_true, self.numpy(y_true)), 0)

    def result(self):
        if self.curve == 'ROC':
            return roc_auc_score(self.y_true, self.y_pred)
        else:
            precision, recall, _ = precision_recall_curve(
                    self.y_true, self.y_pred)
            return auc(recall, precision)