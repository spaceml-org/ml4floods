
from typing import List, Dict, Any, Callable, Optional
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from collections import OrderedDict



@torch.no_grad()
def compute_confusions(ground_truth_outputs: torch.Tensor,
                       test_outputs_categorical: torch.Tensor, num_class: int,
                       remove_class_zero:bool=False) -> torch.Tensor:
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

def binary_accuracy(cm_agg)->float:
    tp = cm_agg[1, 1]
    tn = cm_agg[0, 0]

    return (tp+tn) / (cm_agg.sum() + 1e-6)

def binary_precision(cm_agg) -> float:
    tp = cm_agg[1, 1]
    fp = cm_agg[1, 0]
    return tp / (tp + fp + 1e-6)

def binary_recall(cm_agg) -> float:
    tp = cm_agg[1, 1]
    fn = cm_agg[0, 1]
    return tp / (tp + fn + 1e-6)


def calculate_iou(confusions: torch.Tensor, labels: List[str]) ->Dict[str, float]:
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


def calculate_recall(confusions: torch.Tensor, labels: List[str]) ->Dict[str, float]:
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_negative = np.sum(conf_matrix, 0) - true_positive
    recall = true_positive / (true_positive + false_negative  + 1e-6)

    recall_dict = {}
    for i, l in enumerate(labels):
        recall_dict[l] = recall[i]
    return recall_dict


def calculate_precision(confusions: torch.Tensor, labels: List[str]) ->Dict[str, float]:
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_positive = np.sum(conf_matrix, 1) - true_positive
    precision = true_positive / (true_positive + false_positive  + 1e-6)

    precision_dict = {}
    for i, l in enumerate(labels):
        precision_dict[l] = precision[i]
    return precision_dict


def plot_metrics(metrics_dict, label_names, thresholds_water = None):
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
    if thresholds_water is not None:
        for i, (rec, prec)  in enumerate(zip(plot_df['Recall'],plot_df['Precision'])):
            ax[0].text(rec, prec, '%10.3f' % thresholds_water[i])
        for i, (fps, tps)  in enumerate(zip(plot_df['FP Rate'],plot_df['TP Rate'])):
            ax[1].text(fps, tps, '%10.3f' % thresholds_water[i])

    sns.lineplot(data=plot_df, x="FP Rate", y="TP Rate", ax=ax[1])

    plt.show()
    
    print("Per Class IOU", json.dumps(metrics_dict['iou'], indent=4, sort_keys=True))
  
def convert_targets_to_v1(target: torch.Tensor) -> torch.Tensor:
    
    clear_clouds = target[:,0,:,:]
    land_water = target[:,1,:,:]
    
    v1gt = land_water.clone() # {0: invalid, 1: land, 2: water}
    v1gt[clear_clouds == 2] = 3

    return v1gt.unsqueeze(dim = 1)
      
def compute_metrics(dataloader:torch.utils.data.dataloader.DataLoader,
                    pred_fun: Callable, thresholds_water=np.arange(0,1,.05),
                    threshold:float=.5,
                    plot=False, mask_clouds:bool=False, convert_targets:bool = True) -> Dict:
    """
    Run inference on a dataloader and compute metrics for that data
    
    Args:
        dataloader: pytorch Dataloader for test set
        pred_fun: function to perform inference using a model
        thresholds_water: list of threshold for precision/recall curves
        threshold: threshold to compute the confusion matrix
        plot: flag for calling plot method with metrics
        mask_clouds: if True this will compute the confusion matrices for "land" and "water" classes only masking
            the cloudy pixels. If mask_clouds the output confusion matrices will be (n_samples, 2, 2) otherwise they will
            be (n_samples, 3, 3)
        convert_targets: if True converts targets from v2 to v1
        
        returns: dictionary of metrics
    """
    confusions = []
    
    # Sort thresholds from high to low
    thresholds_water = np.sort(thresholds_water)
    thresholds_water = thresholds_water[-1::-1]
    confusions_thresh = []

    # This is constant: we're using this class convention to compute the PR curve
    if mask_clouds:
        num_class, label_names = 2, ["land", "water"]
    else:
        num_class, label_names = 3, ["land", "water", "cloud"]

    for i, batch in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        test_inputs, ground_truth = batch["image"], batch["mask"]

        test_outputs = pred_fun(test_inputs)
        
        if convert_targets:
            ground_truth = convert_targets_to_v1(ground_truth)
            
        if mask_clouds:
            # assert test_outputs.shape[1] == 1, f"Mode mask clouds expects 1 channel output image found {test_outputs.shape}"
            if test_outputs.shape[1] == 3:    
                test_outputs_categorical = test_outputs[:,1] > threshold
                probs_water_pr_curve = test_outputs[:, 1]
            else:
                test_outputs_categorical = test_outputs[:, 0] > threshold
                probs_water_pr_curve = test_outputs[:, 0]
        else:
            assert test_outputs.shape[1] == num_class, f"Mode normal expects {num_class} channel output image found {test_outputs.shape}"
            test_outputs_categorical = torch.argmax(test_outputs, dim=1).long()
            probs_water_pr_curve = test_outputs[:, 1]

        # Ground truth version v1 is 1 channel tensor 3-classes (B, 1, H, W)  {0: invalid, 1: land, 2: water, 3: cloud}
        # Ground truth version v2 is 2 channel tensor 2-classes (B, 2, H, W) [{0: invalid, 1: land, 2: cloud}, {0: invalid, 1: land, 2: water}]
        if ground_truth.shape[1] > 1:
            # Version v2
            assert mask_clouds, f"Expected ground truth of one band found {ground_truth.shape}"
            water_ground_truth = ground_truth[:, 1] # (batch_size, H, W)
            invalids = (water_ground_truth == 0).to(test_outputs_categorical.device) # (batch_size, H, W)
            ground_truth_outputs = torch.clone(water_ground_truth).to(test_outputs_categorical.device)
        else:
            # Version v1
            ground_truth_outputs = torch.clone(ground_truth[:, 0].to(test_outputs_categorical.device))
            # Save invalids to discount
            invalids = ground_truth_outputs == 0

            # Do not consider cloud pixels for metrics
            if mask_clouds:
                invalids |= (ground_truth_outputs == 3)

        ground_truth_outputs[invalids] = 1 # (batch_size, H, W)
        ground_truth_outputs -= 1
        
        # Set invalids in pred to zero
        test_outputs_categorical[invalids] = 0  # (batch_size, H, W)

        confusions_batch = compute_confusions(ground_truth_outputs, test_outputs_categorical,
                                              num_class=num_class, remove_class_zero=False)
        # confusions_batch is (batch_size, num_class, num_class)

        # Discount invalids
        inv_substract = torch.sum(invalids, dim=(1, 2)).to(confusions_batch.device) # (batch_size, )

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
        for thr in thresholds_water:
            # keep invalids in pred to zero
            test_outputs_categorical_thresh[valids & (probs_water_pr_curve > thr)] = 1

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



def group_confusion(confusions:torch.Tensor, cems_code:np.ndarray,fun:Callable,
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

def compute_metrics_v2(dataloader:torch.utils.data.dataloader.DataLoader,
                       pred_fun: Callable, thresholds_water:Optional[np.array]=None,
                       threshold_water:float=.5, threshold_clouds=.5,
                       plot=False, mask_clouds:bool=True) -> Dict:
    """
    Run inference on a dataloader and compute metrics for that data
    
    Args:
        dataloader: pytorch Dataloader for test set
        pred_fun: function to perform inference using a model
        thresholds_water: list of threshold for precision/recall curves
        threshold_water: threshold of water to compute the confusion matrix
        threshold_clouds: threshold of clouds to compute the confusion matrix
        plot: flag for calling plot method with metrics
        mask_clouds: if True this will compute the confusion matrices for "land" and "water" classes only masking
            the cloudy pixels. If mask_clouds the output confusion matrices will be (n_samples, 2, 2) otherwise they will
            be (n_samples, 3, 3)

        
        returns: dictionary of metrics
    """
    confusions = []
    
    # Sort thresholds from high to low
    if not thresholds_water:
        thresholds_water = [0, 1e-3, 1e-2] + np.arange(0.05, .96, .05).tolist() + [.99, .995, .999]
        thresholds_water = np.array(thresholds_water)

    # thresholds_water sorted from high to low
    thresholds_water = np.sort(thresholds_water)
    thresholds_water = thresholds_water[-1::-1]
    confusions_thresh = []

    # This is constant: we're using this class convention to compute the PR curve
    if mask_clouds:
        num_class, label_names = 2, ["land", "water"]
    else:
        num_class, label_names = 3, ["land", "water", "cloud"]

    for i, batch in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        test_inputs, ground_truth = batch["image"], batch["mask"]

        test_outputs = pred_fun(test_inputs)
      
        test_outputs_categorical = test_outputs[:,1] > threshold_water
        probs_water_pr_curve = test_outputs[:, 1]
        water_ground_truth = ground_truth[:, 1] # (batch_size, H, W)
        invalids = (water_ground_truth == 0).to(test_outputs_categorical.device) # (batch_size, H, W)
        ground_truth_outputs = torch.clone(water_ground_truth).to(test_outputs_categorical.device)
        
        if not mask_clouds:
            assert test_outputs.shape[1] == 2, f"Mode mask clouds expects 2 channel output image found {test_outputs.shape}"
            test_outputs_categorical[test_outputs[:, 0] > threshold_clouds] = 2
            water_ground_truth[ground_truth[:, 0] == 2] = 3 # (batch_size, H, W)
            invalids |= (ground_truth[:, 0] == 0).to(test_outputs_categorical.device)

        # Ground truth version v2 is 2 channel tensor 2-classes (B, 2, H, W) [{0: invalid, 1: land, 2: cloud}, {0: invalid, 1: land, 2: water}]

        ground_truth_outputs[invalids] = 1 # (batch_size, H, W)
        ground_truth_outputs -= 1
        
        # Set invalids in pred to zero
        test_outputs_categorical[invalids] = 0  # (batch_size, H, W)

        confusions_batch = compute_confusions(ground_truth_outputs, test_outputs_categorical,
                                              num_class=num_class, remove_class_zero=False)
        # confusions_batch is (batch_size, num_class, num_class)

        # Discount invalids
        inv_substract = torch.sum(invalids, dim=(1, 2)).to(confusions_batch.device) # (batch_size, )

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
        for thr in thresholds_water:
            # keep invalids in pred to zero
            test_outputs_categorical_thresh[valids & (probs_water_pr_curve > thr)] = 1

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


def compute_positives(ground_truth_outputs: torch.Tensor,
                       water_outputs_categorical: torch.Tensor, convert_targets:bool = True) -> torch.Tensor:
    
    """
    Computes FP, FN and TP given a pair of ground truth and water/land predictions 
    
    """
    
    ground_truth = ground_truth_outputs.clone().type(torch.uint8) 
    
    if convert_targets:
        ground_truth = convert_targets_to_v1(ground_truth).squeeze(0)
    
    invalids = ground_truth == 0  # (batch_size, H, W) gpu
    cloudy = ground_truth == 3

    # Set invalids in pred to zero

    water_outputs = water_outputs_categorical.clone().type(torch.uint8) 
    water_outputs[invalids] = 0  # (batch_size, H, W)
    
    FP = (water_outputs == 2) & (ground_truth == 1) #preds water is land
    FN = (water_outputs == 1) & (ground_truth == 2) #preds land is water
    TP = (water_outputs == 2) & (ground_truth == 2) # pred ok!
    
    positives = torch.zeros(size = ground_truth.shape)
    positives[invalids] = 4
    positives[cloudy] = 4
    positives[FP] = 1
    positives[FN] = 2
    positives[TP] = 3

    return positives.squeeze(0)
        

