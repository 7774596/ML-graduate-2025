import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

class XGBoostTree:
    def __init__(self, max_depth=6, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, n_bins=32):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        
        # Tree structure (lists for fast python access during inference)
        self.feature = []
        self.threshold = []
        self.children_left = []
        self.children_right = []
        self.value = []
        
    def _calc_leaf_weight(self, G, H):
        return -G / (H + self.reg_lambda)

    def _find_best_split(self, feat_idx, X_binned, indices, g, h, G_total, H_total):
        x_feat = X_binned[indices, feat_idx]
        
        # Build histograms
        G_hist = np.bincount(x_feat, weights=g[indices], minlength=self.n_bins + 1)
        H_hist = np.bincount(x_feat, weights=h[indices], minlength=self.n_bins + 1)
        
        G_L = np.cumsum(G_hist)
        H_L = np.cumsum(H_hist)
        
        G_R = G_total - G_L
        H_R = H_total - H_L
        
        # Vectorized gain calculation
        valid_mask = (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
        
        if not np.any(valid_mask):
            return -float('inf'), feat_idx, -1
            
        gains = 0.5 * ((G_L**2 / (H_L + self.reg_lambda)) + 
                       (G_R**2 / (H_R + self.reg_lambda)) - 
                       (G_total**2 / (H_total + self.reg_lambda))) - self.gamma
        
        gains[~valid_mask] = -float('inf')
        
        current_best_bin = np.argmax(gains)
        current_max_gain = gains[current_best_bin]
        
        return current_max_gain, feat_idx, current_best_bin

    def fit(self, X_binned, g, h, bin_thresholds):
        n_samples, n_features = X_binned.shape
        # Array to store prediction for each sample (optimization for training speed)
        sample_predictions = np.zeros(n_samples)
        
        # Stack for DFS: (node_idx, sample_indices, depth)
        stack = [(0, np.arange(n_samples), 0)]
        
        # Initialize root
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.children_left.append(-1)
        self.children_right.append(-1)
        self.value.append(0.0)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            while stack:
                node_idx, indices, depth = stack.pop()
                
                G_total = np.sum(g[indices])
                H_total = np.sum(h[indices])
                
                # Check stopping criteria
                if depth >= self.max_depth or len(indices) < 2 or H_total < self.min_child_weight:
                    leaf_val = self._calc_leaf_weight(G_total, H_total)
                    self.value[node_idx] = leaf_val
                    # Optimization: Record prediction for training samples in this leaf
                    sample_predictions[indices] = leaf_val
                    continue
                    
                best_gain = -float('inf')
                best_feat = -1
                best_bin_idx = -1
                
                # Histogram-based split finding
                # Optimization: Only loop through features if we have enough samples
                # (Already checked len(indices) < 2)
                
                # Parallelize feature loop
                if len(indices) > 500: # Only parallelize for large enough nodes to amortize overhead
                    futures = []
                    for feat_idx in range(n_features):
                        futures.append(executor.submit(self._find_best_split, feat_idx, X_binned, indices, g, h, G_total, H_total))
                    
                    for f in futures:
                        gain, feat_idx, bin_idx = f.result()
                        if gain > best_gain:
                            best_gain = gain
                            best_feat = feat_idx
                            best_bin_idx = bin_idx
                else:
                    # Serial execution for small nodes
                    for feat_idx in range(n_features):
                        gain, feat_idx, bin_idx = self._find_best_split(feat_idx, X_binned, indices, g, h, G_total, H_total)
                        if gain > best_gain:
                            best_gain = gain
                            best_feat = feat_idx
                            best_bin_idx = bin_idx
                
                if best_gain > 0:
                    split_val = bin_thresholds[best_feat][best_bin_idx] if len(bin_thresholds[best_feat]) > best_bin_idx else best_bin_idx
                    
                    x_feat_best = X_binned[indices, best_feat]
                    left_mask = x_feat_best <= best_bin_idx
                    left_indices = indices[left_mask]
                    right_indices = indices[~left_mask]
                    
                    self.feature[node_idx] = best_feat
                    self.threshold[node_idx] = split_val
                    
                    # Create children
                    left_node_idx = len(self.feature)
                    self.feature.append(-1)
                    self.threshold.append(0.0)
                    self.children_left.append(-1)
                    self.children_right.append(-1)
                    self.value.append(0.0)
                    
                    right_node_idx = len(self.feature)
                    self.feature.append(-1)
                    self.threshold.append(0.0)
                    self.children_left.append(-1)
                    self.children_right.append(-1)
                    self.value.append(0.0)
                    
                    self.children_left[node_idx] = left_node_idx
                    self.children_right[node_idx] = right_node_idx
                    
                    stack.append((right_node_idx, right_indices, depth + 1))
                    stack.append((left_node_idx, left_indices, depth + 1))
                else:
                    leaf_val = self._calc_leaf_weight(G_total, H_total)
                    self.value[node_idx] = leaf_val
                    sample_predictions[indices] = leaf_val
        
        return sample_predictions

    def predict_single(self, sample, feature_names, encoders):
        # Fast path for single sample inference with on-demand feature extraction
        node_idx = 0
        # Use local variables for speed
        feature = self.feature
        threshold = self.threshold
        children_left = self.children_left
        children_right = self.children_right
        value = self.value
        
        while True:
            feat_idx = feature[node_idx]
            if feat_idx == -1:
                return value[node_idx]
            
            # On-demand extraction
            name = feature_names[feat_idx]
            val = sample.get(name, 0)
            
            # Check encoder
            encoder = encoders.get(name)
            if encoder is not None:
                val = encoder.get(str(val), 0)
            
            if type(val) is float:
                pass
            elif type(val) is int:
                val = float(val)
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
            
            if val <= threshold[node_idx]:
                node_idx = children_left[node_idx]
            else:
                node_idx = children_right[node_idx]

class Model:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1, reg_lambda=1.0, gamma=0.1, min_child_weight=1.0, n_bins=32):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.trees = []
        self.bin_thresholds = []
        self.base_score = 0.0

    def _sigmoid(self, z):
        if z > 40: return 1.0
        if z < -40: return 0.0
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y, learning_rate=None, epochs=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epochs is not None:
            self.n_estimators = epochs

        n_samples, n_features = X.shape
        self.bin_thresholds = []
        X_binned = np.zeros_like(X, dtype=np.uint8)
        
        # Fast binning with subsampling
        subsample_size = min(10000, n_samples)
        subsample_idx = np.random.choice(n_samples, subsample_size, replace=False)
        
        for i in range(n_features):
            # Estimate thresholds from subsample
            percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            thresholds = np.percentile(X[subsample_idx, i], percentiles)
            thresholds = np.unique(thresholds) # Remove duplicates
            self.bin_thresholds.append(thresholds)
            X_binned[:, i] = np.searchsorted(thresholds, X[:, i])

        p = np.mean(y)
        self.base_score = np.log(p / (1 - p))
        preds = np.full(n_samples, self.base_score)

        for i in range(self.n_estimators):
            p_pred = 1 / (1 + np.exp(-preds)) # Use numpy for batch
            g = p_pred - y
            h = p_pred * (1 - p_pred)
            
            tree = XGBoostTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda, 
                               gamma=self.gamma, min_child_weight=self.min_child_weight, n_bins=self.n_bins)
            # Tree fit now returns predictions for X, avoiding re-traversal
            tree_preds = tree.fit(X_binned, g, h, self.bin_thresholds)
            self.trees.append(tree)
            
            preds += self.learning_rate * tree_preds

    def predict_single(self, sample, feature_names, encoders):
        # Optimized for single sample inference
        pred = self.base_score
        
        # Unroll loop for single tree case
        if len(self.trees) == 1:
            pred += self.learning_rate * self.trees[0].predict_single(sample, feature_names, encoders)
        else:
            for tree in self.trees:
                pred += self.learning_rate * tree.predict_single(sample, feature_names, encoders)
                
        return self._sigmoid(pred)

    def predict_proba(self, X):
        # Fallback for batch prediction (if needed)
        # Not optimized as much as single prediction because the task emphasizes latency (single sample)
        # But we can implement a simple loop
        n_samples = X.shape[0]
        preds = np.full(n_samples, self.base_score)
        
        for i in range(n_samples):
            for tree in self.trees:
                preds[i] += self.learning_rate * tree.predict_single(X[i])
                
        return 1 / (1 + np.exp(-preds))

    def predict(self, X):
        # This is rarely used in the evaluation loop directly, 
        # but if it is, it uses the batch predict_proba
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)