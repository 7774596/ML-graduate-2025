import pandas as pd
import numpy as np
import os
import pickle
from model import Model
from itertools import combinations

class Solution:
    def __init__(self):
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
        self.te_inter_pairs = None
        self.winsor = {}
        self.poly_top_k = 8

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
            self.te_inter_pairs = obj.get('te_inter_pairs', None)
            self.winsor = obj.get('winsor', {})
            self.poly_top_k = obj.get('poly_top_k', 8)

    def _ohe_one(self, sample):
        blocks = []
        for c, info in self.encoders.items():
            cats = info['categories']
            v = str(sample.get(c, ''))
            row = [1.0 if v == cat else 0.0 for cat in cats]
            blocks.append(row)
        return np.hstack(blocks) if blocks else np.empty((0,))

    def _num_one(self, sample):
        vals = []
        for c in self.numeric_cols:
            # ============ 新增特征对应逻辑 ============
            if c == 'job_edu_match':
                job = str(sample.get('job', ''))
                edu = str(sample.get('education', ''))
                high_edu_jobs = ['management', 'technician', 'admin.']
                vals.append(1.0 if (job in high_edu_jobs and edu == 'tertiary') else 0.0)
            elif c == 'has_negative_balance':
                bal = float(sample.get('balance', 0.0))
                vals.append(1.0 if bal < 0 else 0.0)
            elif c == 'has_any_debt':
                loan = str(sample.get('loan', 'no'))
                housing = str(sample.get('housing', 'no'))
                vals.append(1.0 if (loan == 'yes' or housing == 'yes') else 0.0)
            elif c == 'total_loans':
                loan = 1.0 if str(sample.get('loan', 'no')) == 'yes' else 0.0
                housing = 1.0 if str(sample.get('housing', 'no')) == 'yes' else 0.0
                vals.append(loan + housing)
            elif c == 'total_contacts':
                camp = float(sample.get('campaign', 0))
                prev = float(sample.get('previous', 0))
                vals.append(camp + prev)
            elif c == 'contact_ratio':
                camp = float(sample.get('campaign', 0))
                prev = float(sample.get('previous', 0))
                total = camp + prev + 1
                vals.append(camp / total)
            elif c == 'contact_intensity':
                camp = float(sample.get('campaign', 0))
                pdays = max(float(sample.get('pdays', -1)), 1)
                vals.append(camp / (pdays + 1))
            elif c == 'is_single':
                vals.append(1.0 if str(sample.get('marital', '')) == 'single' else 0.0)
            elif c == 'is_divorced':
                vals.append(1.0 if str(sample.get('marital', '')) == 'divorced' else 0.0)
            elif c == 'prev_success':
                vals.append(1.0 if str(sample.get('poutcome', '')) == 'success' else 0.0)
            elif c == 'balance_sqrt':
                v = float(sample.get('balance', 0.0))
                lb, ub = self.winsor.get('balance', (None, None))
                if lb is not None and ub is not None:
                    v = min(max(v, lb), ub)
                vals.append(np.sign(v) * np.sqrt(abs(v)))
            elif c == 'duration_sqrt':
                v = float(sample.get('duration', 0.0))
                lb, ub = self.winsor.get('duration', (None, None))
                if lb is not None and ub is not None:
                    v = min(max(v, lb), ub)
                vals.append(np.sqrt(max(v, 0)))
            elif c == 'balance_duration_inter':
                bal = float(sample.get('balance', 0.0))
                dur = float(sample.get('duration', 0.0))
                lb_b, ub_b = self.winsor.get('balance', (None, None))
                lb_d, ub_d = self.winsor.get('duration', (None, None))
                if lb_b and ub_b: bal = min(max(bal, lb_b), ub_b)
                if lb_d and ub_d: dur = min(max(dur, lb_d), ub_d)
                bl = np.sign(bal) * np.log1p(abs(bal))
                dl = np.sign(dur) * np.log1p(abs(dur))
                vals.append(bl * dl)
            elif c == 'balance_debt_ratio':
                bal = float(sample.get('balance', 0.0))
                loan = str(sample.get('loan', 'no'))
                housing = str(sample.get('housing', 'no'))
                has_debt = 1.0 if (loan == 'yes' or housing == 'yes') else 0.0
                lb, ub = self.winsor.get('balance', (None, None))
                if lb and ub: bal = min(max(bal, lb), ub)
                vals.append(bal / (has_debt + 1))
            
            # ============ 原有特征逻辑 ============
            elif c == 'prev_contacted':
                pdays = float(sample.get('pdays', -1))
                vals.append(1.0 if pdays >= 0 else 0.0)
            elif c == 'balance_log' or c == 'balance_log2':
                v = float(sample.get('balance', 0.0))
                lb, ub = self.winsor.get('balance', (None, None))
                if lb and ub: v = min(max(v, lb), ub)
                bl = np.sign(v) * np.log1p(abs(v))
                vals.append(bl if c == 'balance_log' else bl * bl)
            elif c == 'duration_log' or c == 'duration_log2':
                v = float(sample.get('duration', 0.0))
                lb, ub = self.winsor.get('duration', (None, None))
                if lb and ub: v = min(max(v, lb), ub)
                dl = np.sign(v) * np.log1p(abs(v))
                vals.append(dl if c == 'duration_log' else dl * dl)
            elif c.startswith('day_'):
                d = float(sample.get('day', 0.0))
                ds = np.sin(2.0 * np.pi * d / 31.0)
                dc = np.cos(2.0 * np.pi * d / 31.0)
                if c == 'day_sin': vals.append(ds)
                elif c == 'day_cos': vals.append(dc)
                else: vals.append(ds * dc)
            elif c == 'campaign_log1p':
                vals.append(np.log1p(max(float(sample.get('campaign', 0)), 0.0)))
            elif c == 'previous_log1p':
                vals.append(np.log1p(max(float(sample.get('previous', 0)), 0.0)))
            elif c == 'pdays_pos_log1p':
                v = float(sample.get('pdays', -1))
                vals.append(np.log1p(max(v, 0.0)))
            # 在_num_one()方法中的else分支之前添加:

            elif c == 'balance_log3':
                v = float(sample.get('balance', 0.0))
                lb, ub = self.winsor.get('balance', (None, None))
                if lb and ub: v = min(max(v, lb), ub)
                bl = np.sign(v) * np.log1p(abs(v))
                vals.append(bl ** 3)

            elif c == 'duration_log3':
                v = float(sample.get('duration', 0.0))
                lb, ub = self.winsor.get('duration', (None, None))
                if lb and ub: v = min(max(v, lb), ub)
                dl = np.sign(v) * np.log1p(abs(v))
                vals.append(dl ** 3)

            elif c == 'duration_reciprocal':
                v = float(sample.get('duration', 0.0))
                lb, ub = self.winsor.get('duration', (None, None))
                if lb and ub: v = min(max(v, lb), ub)
                dl = np.sign(v) * np.log1p(abs(v))
                vals.append(1.0 / (abs(dl) + 0.1))

            elif c == 'balance_duration_ratio':
                bal = float(sample.get('balance', 0.0))
                dur = float(sample.get('duration', 0.0))
                lb_b, ub_b = self.winsor.get('balance', (None, None))
                lb_d, ub_d = self.winsor.get('duration', (None, None))
                if lb_b and ub_b: bal = min(max(bal, lb_b), ub_b)
                if lb_d and ub_d: dur = min(max(dur, lb_d), ub_d)
                bl = np.sign(bal) * np.log1p(abs(bal))
                dl = np.sign(dur) * np.log1p(abs(dur))
                vals.append(bl / (abs(dl) + 0.1))

            elif c == 'camp_dur_product':
                camp = float(sample.get('campaign', 0))
                dur = float(sample.get('duration', 0.0))
                lb, ub = self.winsor.get('duration', (None, None))
                if lb and ub: dur = min(max(dur, lb), ub)
                dl = np.sign(dur) * np.log1p(abs(dur))
                vals.append(camp * dl)

            elif c == 'balance_per_contact':
                bal = float(sample.get('balance', 0.0))
                camp = float(sample.get('campaign', 0))
                lb, ub = self.winsor.get('balance', (None, None))
                if lb and ub: bal = min(max(bal, lb), ub)
                vals.append(bal / (camp + 1))

            # 月份特征
            elif c == 'month_sin' or c == 'month_cos':
                month_map = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 
                    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month_num = month_map.get(str(sample.get('month', '')), 0)
                if c == 'month_sin':
                    vals.append(np.sin(2 * np.pi * month_num / 12))
                else:
                    vals.append(np.cos(2 * np.pi * month_num / 12))

            elif c == 'is_summer':
                vals.append(1.0 if str(sample.get('month', '')) in ['jun', 'jul', 'aug'] else 0.0)
            elif c == 'is_year_end':
                vals.append(1.0 if str(sample.get('month', '')) in ['nov', 'dec'] else 0.0)
            elif c == 'is_q1':
                vals.append(1.0 if str(sample.get('month', '')) in ['jan', 'feb', 'mar'] else 0.0)

            elif c == 'is_month_start':
                day = float(sample.get('day', 0))
                vals.append(1.0 if day <= 5 else 0.0)
            elif c == 'is_month_end':
                day = float(sample.get('day', 0))
                vals.append(1.0 if day >= 26 else 0.0)
            else:
                try:
                    vals.append(float(sample.get(c, 0.0)))
                except:
                    vals.append(0.0)
        return np.array(vals, dtype=float)

    def _te_one(self, sample):
        if not self.te_maps or not self.te_names:
            return np.empty((0,), dtype=float)
        vals = []
        mu0 = next(iter(self.te_maps.values()))['global_mean'] if self.te_maps else 0.0
        for name in self.te_names:
            field = name.replace('te_', '')
            info = self.te_maps.get(field, None)
            if info is None:
                vals.append(float(mu0))
                continue
            v = str(sample.get(field, ''))
            mu = info['global_mean']
            te_map = info['map']
            vals.append(float(te_map.get(v, mu)))
        return np.array(vals, dtype=float)

    def _te_inter_one(self, te_vec):
        if te_vec.size == 0 or not self.te_inter_pairs or not self.te_names:
            return np.empty((0,), dtype=float)
        name2idx = {n: i for i, n in enumerate(self.te_names)}
        vals = []
        for a, b in self.te_inter_pairs:
            ia = name2idx.get(a, None)
            ib = name2idx.get(b, None)
            if ia is None or ib is None:
                continue
            vals.append(float(te_vec[ia] * te_vec[ib]))
        return np.array(vals, dtype=float)

    def _poly_one(self, x_sel):
        """添加多项式特征"""
        n_features = min(self.poly_top_k, x_sel.shape[0])
        poly_vals = []
        for i, j in combinations(range(n_features), 2):
            poly_vals.append(x_sel[i] * x_sel[j])
        if poly_vals:
            return np.concatenate([x_sel, np.array(poly_vals)])
        return x_sel

    def forward(self, sample: dict) -> dict:
        if self.encoders is not None and self.numeric_cols is not None:
            X_cat = self._ohe_one(sample)
            X_num = self._num_one(sample)
            X_te = self._te_one(sample)
            X_te_inter = self._te_inter_one(X_te)
            
            parts = [X_cat, X_num]
            if X_te.size: parts.append(X_te)
            if X_te_inter.size: parts.append(X_te_inter)
            
            x_full = np.hstack(parts)
            x_sel = x_full[self.keep_idx] if self.keep_idx is not None else x_full
            
            # 添加多项式特征
            x_with_poly = self._poly_one(x_sel)
            
            X_scaled = (x_with_poly - self.X_mean) / self.X_std
            X_scaled = X_scaled.reshape(1, -1)
        else:
            raise ValueError("模型未正确初始化")
        
        pred = self.model.predict(X_scaled)[0]
        
        # 裁剪到年龄范围
        if self.y_min is not None and self.y_max is not None:
            pred = float(min(max(pred, self.y_min), self.y_max))
        
        return {'prediction': float(pred)}