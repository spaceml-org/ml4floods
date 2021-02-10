import pandas as pd
import numpy as np
import os
import json
from utils import fsutils

CLASS_NAMES = {0: "land", 1: "water", 2: "cloud"}


def save_results(cm, filename, verbose=True, num_class=3, extra={}):
    dats_2_pd = compute_metrics_confusion(cm)

    # Print stats
    if verbose:
        _ = produce_tables(dats_2_pd, num_class, cm, verbose=verbose)


    # Save results
    json_old = dats_json_old(dats_2_pd, num_class)
    json_old["union_class"] = fsutils.castdict(json_old["union_class"])
    json_old["intersection_class"] = fsutils.castdict(json_old["intersection_class"])
    json_old["total_class"] = fsutils.castdict(json_old["total_class"])
    json_old["confusion_matrix"] = [fsutils.castmatrix(c) for c in cm.tolist()]
    json_old["total"] = fsutils.cast2(json_old["total"])
    json_old["correct"] = fsutils.cast2(json_old["correct"])
    json_old.update(extra)

    with open(filename, "w") as fh:
        json.dump(json_old, fh)


def compute_metrics_confusion(cm):
    num_classes = cm.shape[1]

    dats_2_pd = {"total": np.sum(cm, axis=(1, 2))}
    for c in range(num_classes):
        dats_2_pd["intersection_%d" % c] = cm[:, c, c]
        if "correct" not in dats_2_pd:
            dats_2_pd["correct"] = dats_2_pd["intersection_%d" % c].copy()
        else:
            dats_2_pd["correct"] += dats_2_pd["intersection_%d" % c]

        dats_2_pd["total_%d" % c] = np.sum(cm[:, :, c], axis=1)

        dats_2_pd["union_%d" % c] = dats_2_pd["intersection_%d" % c].copy()
        for cother in range(num_classes):
            if cother != c:
                dats_2_pd["union_%d" % c] += cm[:, c, cother] + cm[:, cother, c]
    return dats_2_pd


def dats_json_old(dats_2_pd,num_classes):
    dats_json = {
        "total": dats_2_pd["total"],
        "correct": dats_2_pd["correct"]
    }
    union = {}
    intersection = {}
    total = {}
    for c in range(num_classes):
        union[c] = dats_2_pd["union_%d"%c]
        intersection[c] = dats_2_pd["intersection_%d" % c]
        total[c] = dats_2_pd["total_%d" % c]

    dats_json["union_class"] = union
    dats_json["intersection_class"] = intersection
    dats_json["total_class"] = total
    return dats_json


def process_json(folder, verbose=True):
    with open(os.path.join(folder, "metrics_worldfloods.json"), "r") as fh:
        dat = json.load(fh)

    cm = np.array(dat["confusion_matrix"])
    num_classes = cm.shape[1]

    dats_2_pd = compute_metrics_confusion(cm)
    
    if verbose:
        print("Model: %s checkpoint: %s" % (dat["model"], dat["model_file"]))
    tb_out = produce_tables(dats_2_pd, num_classes, cm, verbose=verbose)
    return tb_out[0], tb_out[1], dat["model_file"]


def process_json_old(folder, verbose=True):
    with open(os.path.join(folder, "metrics_worldfloods.json"), "r") as fh:
        dat = json.load(fh)

    num_classes = 3
    dats_2_pd = {"total": dat["total"],
                 "correct": dat["correct"],
                 "union_0": dat["union_class"]["0"],
                 "union_1": dat["union_class"]["1"],
                 "union_2": dat["union_class"]["2"],
                 "intersection_0": dat["intersection_class"]["0"],
                 "intersection_1": dat["intersection_class"]["1"],
                 "intersection_2": dat["intersection_class"]["2"],
                 "total_0": dat["total_class"]["0"],
                 "total_1": dat["total_class"]["1"],
                 "total_2": dat["total_class"]["2"],
                 }
    cm = np.array(dat["confusion_matrix"])
    print("Model: %s checkpoint: %s" % (dat["model"], dat["model_file"]))
    tb_out = produce_tables(dats_2_pd, num_classes, cm, verbose=verbose)
    return tb_out[0], tb_out[1],dat["model_file"]


def produce_tables(dats_2_pd, num_classes, cm, verbose=True):
    pd_stats = pd.DataFrame(dats_2_pd)
    pd_stats_added = pd_stats.sum(axis=0)

    assert np.all(pd_stats["correct"] == np.sum(pd_stats[["intersection_%d" % c for c in range(num_classes)]],
                                                axis=1)), "Correct not well computed"
    assert np.all(pd_stats["total"] == np.sum(pd_stats[["total_%d" % c for c in range(num_classes)]],
                                              axis=1)), "Total not well computed"

    # Stats per file
    pd_stats["Accuracy"] = pd_stats["correct"] / pd_stats["total"]
    for c in range(num_classes):
        pd_stats["IoU_%d" % c] = pd_stats["intersection_%d" % c] / (pd_stats["union_%d" % c] + 1e-6)
        pd_stats["Accuracy_%d" % c] = pd_stats["intersection_%d" % c] / (pd_stats["total_%d" % c] + 1e-6)
        pd_stats["frac_%d" % c] = pd_stats["total_%d" % c] / pd_stats["total"]
        pd_stats.loc[pd_stats["total_%d" % c] == 0, "IoU_%d" % c] = np.NaN
        pd_stats.loc[pd_stats["total_%d" % c] == 0, "Accuracy_%d" % c] = np.NaN

    pd_stats["mIoU"] = pd_stats[["IoU_%d" % c for c in range(num_classes)]].mean(axis=1)
    # print("mean Accuracy (mean taken over images) %.2f%%"%(pd_stats["Accuracy"].mean()*100))
    # print("mean mIoU (mean taken over images) %.2f%%"%(pd_stats["mIoU"].mean()*100))

    # Stats all images
    for c in range(num_classes):
        cname = CLASS_NAMES[c]
        pd_stats_added["Accuracy_%d" % c] = pd_stats_added["intersection_%d" % c] / pd_stats_added["total_%d" % c]
        pd_stats_added["IoU_%d" % c] = pd_stats_added["intersection_%d" % c] / pd_stats_added["union_%d" % c]
        pd_stats_added["frac_%d" % c] = pd_stats_added["total_%d" % c] / pd_stats_added["total"]
        if verbose:
            print("Acc %s: %.2f%%" % (cname, pd_stats_added["Accuracy_%d" % c] * 100))
            print("IoU %s: %.2f%%" % (cname, pd_stats_added["IoU_%d" % c] * 100))
            print("Frac %s: %.2f%%" % (cname, pd_stats_added["frac_%d" % c] * 100))

    pd_stats_added["Accuracy"] = pd_stats_added["correct"] / pd_stats_added["total"]
    pd_stats_added["mIoU"] = pd_stats_added[["IoU_%d" % c for c in range(num_classes)]].mean()
    pd_stats_added["confusion_matrix"] = cm.sum(axis=0) / cm.sum()
    if verbose:
        print("Accuracy total: %.2f%%" % (pd_stats_added["Accuracy"] * 100))
        print("mIoU total: %.2f%%" % (pd_stats_added["mIoU"] * 100))
    return pd_stats, pd_stats_added