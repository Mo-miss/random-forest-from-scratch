"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    # Step 1: 边界处理
    if len(labels) == 0:
        return 0.0
    unique_vals, counts = np.unique(labels, return_counts=True)
    if len(unique_vals) == 1:
        return 0.0
    
    # Step2 计算各类占比
    total = len(labels)
    p = counts / total
    
    # Step3 Gini公式: G = 1 - sum(p^2)
    gini = 1 - np.sum(p ** 2)
    
    # 返回原生Python float
    return float(gini)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    l_mask = features[:,feature_index] <= threshold 
    r_mask = features[:,feature_index] > threshold 
    lx = features[l_mask]
    ly = labels[l_mask]
    rx = features[r_mask]
    ry = labels[r_mask] 
    return lx,ly,rx,ry

# Step 3 - split_score (not yet solved)
# TODO: implement

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

