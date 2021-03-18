
from typing import List, Dict, Any, Callable
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from collections import OrderedDict


@torch.no_grad()
def compute_confusions(ground_truth_outputs: torch.Tensor, test_outputs_categorical: torch.Tensor, num_class: int,
                       remove_class_zero=False) -> torch.Tensor:
    """
    Compute the confusion matrix for a pair of ground truth and predictions. Returns one confusion matrix for each
    element in the batch

    Args:
        ground_truth_outputs: (B,H,W) tensor with discrete values in [0, num_class] if remove_class_zero else [0,num_class-1]
        test_outputs_categorical: (B,H,W) tensor with discrete values in [0,num_class-1] (expected output of torch.argmax())
        num_class: Number of classes. Must be equal to number of channels of test_outputs_categorical
        remove_class_zero: if true the value zero in ground_truth_outputs is considered a masked value;
        thus removed from the final prediction.

    Returns:
        (B,num_class,num_class) torch.Tensor with a confusion matrix for each image in the batch
        cm[a, b, c] is the number of elements predicted `b` that belonged to class `c` in the image `a` of the batch

    """
    ground_truth = ground_truth_outputs.clone()

    if remove_class_zero:
        # Save invalids to discount
        invalids = ground_truth == 0  # (batch_size, H, W) gpu
        ground_truth[invalids] = 1
        ground_truth -= 1
    
        # Set invalids in pred to zero
        test_outputs_categorical = test_outputs_categorical.clone()
        test_outputs_categorical[invalids] = 0  # (batch_size, H, W)

    confusions_batch = torch.zeros(size=(ground_truth.shape[0], num_class, num_class),
                                   dtype=torch.long)
    for c in range(num_class):
        gtc = (ground_truth == c)
        for c1 in range(num_class):
            pred_c1 = (test_outputs_categorical == c1)
            confusions_batch[:, c1, c] = (pred_c1 & gtc).sum(dim=(1, 2))
    
    if remove_class_zero:
        inv_substract = torch.sum(invalids, dim=(1, 2)).to(confusions_batch.device)
        confusions_batch[:, 0, 0] -= inv_substract
    
    return confusions_batch


def cm_analysis(cm: np.ndarray, labels: List[int], figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.

    Args:
      cm:  Array with shape (batch_size, len(labels), len(labels))
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      figsize:   the size of the figure plotted.
    """
    
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)    
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Predicted'
    cm.columns.name = 'Actual'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()

def binary_accuracy(cm_agg):
    tp = cm_agg[1, 1]
    tn = cm_agg[0, 0]

    return (tp+tn) / cm_agg.sum()

def binary_precision(cm_agg):
    tp = cm_agg[1, 1]
    fp = cm_agg[1, 0]
    return tp / (tp + fp + 1e-6)

def binary_recall(cm_agg):
    tp = cm_agg[1, 1]
    fn = cm_agg[0, 1]
    return tp / (tp + fn + 1e-6)


def calculate_iou(confusions, labels):
    """
    Caculate IoU for a list of confusion matrices 
    
    Args:
        confusions: List with shape (batch_size, len(labels), len(labels))
        labels: List of class names
        
        returns: dictionary of class names and iou scores for that class (summed across whole matrix list)
    """
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_negative = np.sum(conf_matrix, 0) - true_positive
    false_positive = np.sum(conf_matrix, 1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative)
    
    iou_dict = {}
    for i, l in enumerate(labels):
        iou_dict[l] = iou[i]
    return iou_dict


def calculate_recall(confusions, labels):
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_negative = np.sum(conf_matrix, 0) - true_positive
    recall = true_positive / (true_positive + false_negative  + 1e-6)

    recall_dict = {}
    for i, l in enumerate(labels):
        recall_dict[l] = recall[i]
    return recall_dict


def calculate_precision(confusions, labels):
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_positive = np.sum(conf_matrix, 1) - true_positive
    precision = true_positive / (true_positive + false_positive  + 1e-6)

    precision_dict = {}
    for i, l in enumerate(labels):
        precision_dict[l] = precision[i]
    return precision_dict


def plot_metrics(metrics_dict, label_names):
    """
    Plot confusion matrix, precision/recall curve,  tp_rate/fp_rate curve, class IoUs
    
    Args:
        metrics_dict: metrics dictionary as output by 'compute_metrics'
        labels_names: list of class label names
        
        returns: None
    """
    confusions = np.array(metrics_dict['confusions']).transpose(0,2,1)
    confusions_thresh = metrics_dict['confusions_thresholded']
    
    cm_analysis(np.sum(np.array(confusions), axis=0), labels=label_names)

    total_conf_thresh = np.sum(np.array(confusions_thresh), axis=1)
    tp_rates = []
    fp_rates = []

    precisions = []
    recalls = []

    for i in range(len(total_conf_thresh)):
        cur_conf = total_conf_thresh[i]

        tp = cur_conf[1,1]
        tn = cur_conf[0,0]
        fp = cur_conf[1,0]
        fn = cur_conf[0,1]

        tp_rate = tp / (tp+fn)
        fp_rate = fp / (fp+tn)

        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

        precision = tp / (tp+fp)
        recall = tp / (tp+fn)

        precisions.append(precision)
        recalls.append(recall)

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    sns.set_style('darkgrid')
    plot_df = pd.DataFrame({"Precision": precisions, "Recall": recalls, "FP Rate": fp_rates, "TP Rate": tp_rates})
    sns.lineplot(data=plot_df, x="Recall", y="Precision", ax=ax[0])
    sns.lineplot(data=plot_df, x="FP Rate", y="TP Rate", ax=ax[1])

    plt.show()
    
    print("Per Class IOU", json.dumps(metrics_dict['iou'], indent=4, sort_keys=True))
        
def compute_metrics(dataloader, pred_fun, num_class, label_names, thresholds_water=np.arange(0,1,.05), plot=False):
    """
    Run inference on a dataloader and compute metrics for that data
    
    Args:
        dataloader: pytorch Dataloader for test set
        pred_fun: function to perform inference using a model
        num_class: number of classes
        label_names: list of label names
        thresholds: list of threshold for precision/recall curves
        plot: flag for calling plot method with metrics
        
        returns: dictionary of metrics
    """
    confusions = []
    
    # Sort thresholds from high to low
    thresholds_water = np.sort(thresholds_water)
    thresholds_water = thresholds_water[-1::-1]
    confusions_thresh = []
    
    for i, batch in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        test_inputs, ground_truth_outputs = batch["image"], batch["mask"].squeeze(1)
        
        test_outputs = pred_fun(test_inputs)
        
        test_outputs_categorical = torch.argmax(test_outputs, dim=1).long()
        ground_truth_outputs = torch.clone(ground_truth_outputs.to(test_outputs_categorical.device))
        
        # Save invalids to discount
        invalids = ground_truth_outputs == 0
        ground_truth_outputs[invalids] = 1
        ground_truth_outputs -= 1
        
        # Set invalids in pred to zero
        test_outputs_categorical[invalids] = 0  # (batch_size, H, W)

        confusions_batch = compute_confusions(ground_truth_outputs, test_outputs_categorical,
                                              num_class=num_class, remove_class_zero=False)
        # confusions_batch is (batch_size, num_class, num_class)

        # Discount invalids
        inv_substract = torch.sum(invalids, dim=(1, 2)).to(confusions_batch.device)

        confusions_batch[:, 0, 0] -= inv_substract
        confusions.extend(confusions_batch.tolist())
        
        # Thresholded version for precision recall curves
        # Set clouds to land
        test_outputs_categorical_thresh = torch.zeros(ground_truth_outputs.shape, dtype=torch.long,
                                                      device=ground_truth_outputs.device)

        ground_truth_outputs[ground_truth_outputs == 2] = 0

        results = []
        valids = ~invalids
        
        # thresholds_water sorted from high to low
        for threshold in thresholds_water:
            # keep invalids in pred to zero
            test_outputs_categorical_thresh[valids & (test_outputs[:, 1] > threshold)] = 1

            confusions_batch = compute_confusions(ground_truth_outputs,
                                                  test_outputs_categorical_thresh, num_class=2, remove_class_zero=False)   # [batch_size, 2, 2]

            # Discount invalids
            confusions_batch[:, 0, 0] -= torch.sum(invalids.to(confusions_batch.device))

            results.append(confusions_batch.numpy())

        # results is [len(thresholds), batch_size, 2, 2]
        confusions_thresh.append(np.stack(results))

    confusions_thresh = np.concatenate(confusions_thresh, axis=1)
    
    iou = calculate_iou(confusions, labels=label_names)
    recall = calculate_recall(confusions, labels=label_names)
    
    out_dict = {
        'confusions': confusions,
        'confusions_thresholded': confusions_thresh,
        'thresholds': thresholds_water,
        'iou': iou,
        "recall": recall
    }
    
    if plot:
        plot_metrics(out_dict, label_names)
    
    return out_dict


def group_confusion(confusions:torch.Tensor, cems_code:str,fun:Callable,
                   label_names:List[str]) ->List[Dict[str, Any]]:
    CMs = OrderedDict({c:[] for c in sorted(np.unique(cems_code))})

    for code, cms in zip(cems_code,confusions):
        CMs[code].append(cms)
    
    data_out = []
    for k, v in CMs.items():
        ious = fun(v, label_names)
        ious["code"] = k
        data_out.append(ious)
    
    return data_out
        