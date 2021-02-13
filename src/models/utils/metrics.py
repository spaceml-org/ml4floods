import torch
import numpy as np


def compute_confusions(ground_truth_outputs: torch.Tensor, test_outputs_categorical: torch.Tensor, num_class: int,
                       remove_class_zero=False) -> torch.Tensor:
    """
    Compute the confusion matrix for a pair of ground truth and predictions. Returns one confusion matrix for each
    element in the batch

    :param ground_truth_outputs: BCHW tensor with discrete values in [0, num_class] if remove_class_zero else [0,num_class-1]
    :param test_outputs_categorical: BCHW tensor with discrete values in [0,num_class-1] (expected output of torch.argmax())
    :param num_class: Number of classes.
    :param remove_class_zero: if true the value zero in ground_truth_outputs is considered a masked value;
    thus removed from the final prediction.

    :return: (B,num_class,num_class) torch.Tensor with a confusion matrix for each image in the batch

    """
    if remove_class_zero:
        # Save invalids to discount
        ground_truth = ground_truth_outputs.clone()
        invalids = ground_truth == 0  # (batch_size, H, W) gpu
        ground_truth[invalids] = 1
        ground_truth -= 1
    
        # Set invalids in pred to zero
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


def cm_analysis(cm, labels, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()