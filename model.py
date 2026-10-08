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

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)
    w_l = len(left_labels) / n
    w_r = len(right_labels) / n
    return impurity(parent_labels) - (w_l * impurity(left_labels) + w_r * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    """
    寻找最优特征+阈值划分
    Args:
        features: np.ndarray shape (n_samples, n_dims)
        labels: np.ndarray shape (n_samples,)
        feature_indices: list，需要遍历的特征索引
    Returns:
        dict: {'feature_index': int|None, 'threshold': float|None, 'score': float}
    """
    # 初始化最优记录
    best = {
        'feature_index': None,
        'threshold': None,
        'score': 0.0
    }

    for fi in feature_indices:
        # 获取该特征列，去重并排序
        col = features[:, fi]
        unique_vals = np.unique(col)
        # 生成相邻中点作为候选阈值
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for t in thresholds:
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)
            # 跳过任意一侧为空的无效划分
            if len(ll) == 0 or len(rl) == 0:
                continue
            # 计算分裂增益
            score = split_score(labels, ll, rl)
            # 更新最优
            if score > best['score']:
                best['feature_index'] = fi
                best['threshold'] = t
                best['score'] = score
    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """
    判断决策树当前节点是否停止分裂，转为叶子节点
    Args:
        labels: np.ndarray, 当前节点样本标签
        depth: int, 当前节点深度（根节点=0）
        max_depth: int, 树允许的最大深度
        min_samples_split: int, 继续分裂所需要的最少样本数
    Returns:
        bool: True → 停止分裂；False → 继续分裂
    """
    # 条件1：节点纯净，所有标签相同
    pure = len(np.unique(labels)) == 1
    # 条件2：深度达到上限
    depth_reach = depth >= max_depth
    # 条件3：样本数量不足
    too_small = len(labels) < min_samples_split
    
    # 任意条件满足，停止分裂
    return pure or depth_reach or too_small

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

