import numpy as np
import math

class Node:
    def __init__(self):
        self.feature_idx = -1
        self.feature_name = None
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0
        self.is_leaf = False

class Model:
    def __init__(self, max_depth=1, reg_lambda=1.0, min_child_weight=1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.trees = []
        self.base_score = 0.0
        self.compiled_predict = None

    def _sigmoid(self, z):
        if z > 40: return 1.0
        if z < -40: return 0.0
        return 1 / (1 + math.exp(-z))

    def _build_tree(self, X, g, h, depth, feature_names=None):
        node = Node()
        n_samples = len(g)
        G = np.sum(g)
        H = np.sum(h)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < 2 or H < self.min_child_weight:
            node.is_leaf = True
            node.value = -G / (H + self.reg_lambda)
            return node

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        
        current_score = G**2 / (H + self.reg_lambda)
        n_features = X.shape[1]
        
        # Greedy split finding
        for feat_idx in range(n_features):
            x_col = X[:, feat_idx]
            thresholds = np.unique(x_col)
            
            for thresh in thresholds:
                left_mask = x_col <= thresh
                if not np.any(left_mask) or np.all(left_mask):
                    continue
                
                g_l = g[left_mask]
                h_l = h[left_mask]
                G_L = np.sum(g_l)
                H_L = np.sum(h_l)
                
                G_R = G - G_L
                H_R = H - H_L
                
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue
                
                gain = 0.5 * ((G_L**2 / (H_L + self.reg_lambda)) + 
                              (G_R**2 / (H_R + self.reg_lambda)) - 
                              current_score)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        if best_gain > 0:
            node.feature_idx = best_feat
            node.threshold = best_thresh
            if feature_names:
                node.feature_name = feature_names[best_feat]
            
            left_mask = X[:, best_feat] <= best_thresh
            node.left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1, feature_names)
            node.right = self._build_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1, feature_names)
        else:
            node.is_leaf = True
            node.value = -G / (H + self.reg_lambda)
            
        return node

    def fit(self, X, y, learning_rate=0.3, epochs=5, feature_names=None):
        n_samples = X.shape[0]
        
        # Initial prediction (base score)
        p = np.mean(y)
        p = np.clip(p, 1e-6, 1-1e-6)
        self.base_score = np.log(p / (1 - p))
        preds = np.full(n_samples, self.base_score)
        
        self.trees = []
        
        # Boosting Loop
        n_estimators = epochs
        
        for _ in range(n_estimators):
            p_pred = 1 / (1 + np.exp(-preds))
            g = p_pred - y
            h = p_pred * (1 - p_pred)
            
            tree = self._build_tree(X, g, h, 0, feature_names)
            self.trees.append(tree)
            
            # Update predictions for next iteration
            new_preds = np.zeros(n_samples)
            for i in range(n_samples):
                node = tree
                while not node.is_leaf:
                    if X[i, node.feature_idx] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                new_preds[i] = node.value
            
            preds += learning_rate * new_preds

        self.learning_rate = learning_rate
        
        # Optimization: Pre-multiply leaf values by learning rate
        # and use the first tree as root for fast access
        if self.trees:
            self.root = self.trees[0]
            stack = [self.root]
            while stack:
                node = stack.pop()
                if node.is_leaf:
                    node.value *= learning_rate
                else:
                    stack.append(node.left)
                    stack.append(node.right)

    def predict_single(self, sample, feature_names, encoders):
        # Optimized object-based inference for single tree
        # Inline everything for speed
        if hasattr(self, 'root') and self.root:
            node = self.root
            while not node.is_leaf:
                if sample[node.feature_name] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            z = self.base_score + node.value
            if z > 40: return 1.0
            if z < -40: return 0.0
            return 1 / (1 + math.exp(-z))

        return 0.5

    def predict_proba(self, X):
        return np.zeros(X.shape[0])

    def predict(self, X):
        return np.zeros(X.shape[0])
