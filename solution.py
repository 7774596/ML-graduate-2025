import pandas as pd
import numpy as np
import os
import pickle
from model import Model

class Solution:
    def __init__(self):
        """
        初始化模型；优先加载训练好的参数与编码
        """
        self.model = Model()
        self.X_mean = None
        self.X_std = None
        self.encoders = None
        self.numeric_cols = None
        self.feature_names_all = None
        self.feature_names = None
        self.keep_idx = None
        self.te_maps = None
        self.te_names = None
        self.y_min = None
        self.y_max = None

        params_path = 'model_params.pkl'
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                obj = pickle.load(f)
            self.model.weights = obj['weights']
            self.model.bias = obj['bias']
            self.X_mean = obj['X_mean']
            self.X_std = obj['X_std']
            self.encoders = obj['encoders']
            self.numeric_cols = obj['numeric_cols']
            self.feature_names_all = obj.get('feature_names_all', None)
            self.feature_names = obj.get('feature_names', None)
            self.keep_idx = obj.get('keep_idx', None)
            self.te_maps = obj.get('te_maps', None)
            self.te_names = obj.get('te_names', None)
            self.y_min = obj.get('y_min', None)
            self.y_max = obj.get('y_max', None)
        else:
            np.random.seed(42)
            n_features = 15
            self.model.weights = np.random.randn(n_features) * 0.01
            self.model.bias = np.random.randn() * 0.01
            self.X_mean = np.zeros(n_features)
            self.X_std = np.ones(n_features)
            self.keep_idx = None

    def _ohe_one(self, sample):
        blocks = []
        for c, info in self.encoders.items():
            cats = info['categories']
            v = str(sample.get(c, ''))
            row = [1.0 if v == cat else 0.0 for cat in cats]
            blocks.append(row)
        X_cat = np.hstack(blocks) if blocks else np.empty((0,))
        return X_cat

    def _num_one(self, sample):
        vals = []
        for c in self.numeric_cols:
            if c == 'prev_contacted':
                pdays = sample.get('pdays', -1)
                try: pdays = float(pdays)
                except: pdays = -1
                vals.append(1.0 if pdays >= 0 else 0.0)
            elif c == 'balance_log':
                v = sample.get('balance', 0.0)
                try: v = float(v)
                except: v = 0.0
                vals.append(np.sign(v) * np.log1p(abs(v)))
            elif c == 'duration_log':
                v = sample.get('duration', 0.0)
                try: v = float(v)
                except: v = 0.0
                vals.append(np.sign(v) * np.log1p(abs(v)))
            elif c == 'day_sin':
                d = sample.get('day', 0.0)
                try: d = float(d)
                except: d = 0.0
                vals.append(np.sin(2.0 * np.pi * d / 31.0))
            elif c == 'day_cos':
                d = sample.get('day', 0.0)
                try: d = float(d)
                except: d = 0.0
                vals.append(np.cos(2.0 * np.pi * d / 31.0))
            else:
                v = sample.get(c, 0.0)
                try:
                    vals.append(float(v))
                except:
                    vals.append(0.0)
        return np.array(vals, dtype=float)

    def _te_one(self, sample):
        if not self.te_maps:
            return np.empty((0,), dtype=float)
        vals = []
        for f, info in self.te_maps.items():
            v = str(sample.get(f, ''))
            mu = info['global_mean']
            te_map = info['map']
            vals.append(float(te_map.get(v, mu)))
        return np.array(vals, dtype=float)

    def forward(self, sample: dict) -> dict:
        """
        单样本推理
        """
        if self.encoders is not None and self.numeric_cols is not None:
            X_cat = self._ohe_one(sample)
            X_num = self._num_one(sample)
            X_te = self._te_one(sample)
            x_full = np.hstack([X_cat, X_num, X_te]).reshape(1, -1)
            x = x_full[:, self.keep_idx] if self.keep_idx is not None else x_full
            X_scaled = (x - self.X_mean) / self.X_std
        else:
            sample_df = pd.DataFrame([sample])
            X = sample_df.iloc[:, 1:]
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.factorize(X[col])[0]
            X = X.values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = (X - self.X_mean) / self.X_std

        pred = self.model.predict(X_scaled)[0]
        # 预测裁剪到训练年龄范围
        if self.y_min is not None and self.y_max is not None:
            pred = float(min(max(pred, self.y_min), self.y_max))
        return {'prediction': float(pred)}
# ...existing code...