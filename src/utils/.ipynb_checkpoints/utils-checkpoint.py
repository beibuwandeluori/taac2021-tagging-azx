import tensorboardX
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score
import torch
import numpy as np
from .average_precision_calculator import AveragePrecisionCalculator


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))

def calculate_metrics(outputs, targets, metric_name='acc'):
    if len(targets.data.numpy().shape) > 1:
        _, targets = torch.max(targets.detach(), dim=1)
    if len(outputs.data.numpy().shape) > 1 and outputs.data.numpy().shape[1] == 1:  # 尾部是sigmoid
        outputs = torch.cat([1-outputs, outputs], dim=1)

    # print(outputs.shape, targets.shape, pred_labels.size())
    if metric_name == 'acc':
        pred_labels = torch.max(outputs, 1)[1]
        # print(targets, pred_labels)
        return accuracy_score(targets.data.numpy(), pred_labels.detach().numpy())
    elif metric_name == 'auc':
        pred_labels = outputs[:, 1]  # 为假的概率
        return roc_auc_score(targets.data.numpy(), pred_labels.detach().numpy())

# ___________________GAP___________________________________
def calculate_gap(predictions, actuals, top_k=20):
  """Performs a local (numpy) calculation of the global average precision.

  Only the top_k predictions are taken for each of the videos.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.
    top_k: How many predictions to use per video.

  Returns:
    float: The global average precision.
  """
  gap_calculator = AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()

def flatten(l):
  """ Merges a list of lists into a single list. """
  return [item for sublist in l for item in sublist]

def top_k_by_class(predictions, labels, k=20):
  """Extracts the top k predictions for each video, sorted by class.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
    are lists of lists of floats. 'true_positives' is a list of scalars. The
    length of the lists are equal to the number of classes. The entries in the
    predictions variable are probability predictions, and
    the corresponding entries in the labels variable are the ground truth for
    those predictions. The entries in 'true_positives' are the number of true
    positives for each class in the ground truth.

  Raises:
    ValueError: An error occurred when the k is not a positive integer.
  """
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  num_classes = predictions.shape[1]
  prediction_triplets= []
  for video_index in range(predictions.shape[0]):
    prediction_triplets.extend(top_k_triplets(predictions[video_index],labels[video_index], k))
  out_predictions = [[] for v in range(num_classes)]
  out_labels = [[] for v in range(num_classes)]
  for triplet in prediction_triplets:
    out_predictions[triplet[0]].append(triplet[1])
    out_labels[triplet[0]].append(triplet[2])
  out_true_positives = [np.sum(labels[:,i]) for i in range(num_classes)]

  return out_predictions, out_labels, out_true_positives

def top_k_triplets(predictions, labels, k=20):
  """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
  (prediction, class) format"""
  m = len(predictions)
  k = min(k, m)
  indices = np.argpartition(predictions, -k)[-k:]
  return [(index, predictions[index], labels[index]) for index in indices]
# ___________________GAP___________________________________


if __name__ == '__main__':

    pass
