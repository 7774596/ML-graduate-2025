import pandas as pd
import numpy as np
import pickle
from model import Model

DATA_PATH = 'train.csv'
OUT_PATH = 'model_params.pkl'
CATEGORICALS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
TE_FIELDS = ['job', 'marital', 'education', 'month', 'contact', 'poutcome', 'housing']  # 目标均值编码字段
TE_M = 20  # m-estimate 平滑系数

def build_encoders(df, cats):
    encoders = {}
    for c in cats:
        vals = df[c].astype(str).fillna('')
        categories = list(pd.Index(vals).unique())  # 按出现顺序
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        encoders[c] = {'mapping': mapping, 'categories': categories}
    return encoders

def ohe_transform(df, encoders):
    feature_blocks = []
    feature_names = []
    for c, info in encoders.items():
        vals = df[c].astype(str).fillna('')
        mapping = info['mapping']
        n_cat = len(mapping)
        idx = vals.map(lambda v: mapping.get(v, None)).values
        M = np.zeros((len(vals), n_cat), dtype=float)
        for i, k in enumerate(idx):
            if k is not None:
                M[i, k] = 1.0
        feature_blocks.append(M)
        feature_names += [f'{c}={cat}' for cat in info['categories']]
    if feature_blocks:
        X_cat = np.hstack(feature_blocks)
    else:
        X_cat = np.empty((len(df), 0), dtype=float)
    return X_cat, feature_names

def signed_log1p(arr):
    arr = pd.to_numeric(arr, errors='coerce').fillna(0.0).astype(float).values
    return np.sign(arr) * np.log1p(np.abs(arr))

def build_features(df):
    df = df.copy()

    # 衍生：是否被联系过
    if 'pdays' in df.columns:
        df['prev_contacted'] = (df['pdays'] >= 0).astype(int)

    # 数值变换：重尾与周期
    if 'balance' in df.columns:
        df['balance_log'] = signed_log1p(df['balance'])
    if 'duration' in df.columns:
        df['duration_log'] = signed_log1p(df['duration'])
    if 'day' in df.columns:
        day = pd.to_numeric(df['day'], errors='coerce').fillna(0.0).astype(float)
        df['day_sin'] = np.sin(2.0 * np.pi * day / 31.0)
        df['day_cos'] = np.cos(2.0 * np.pi * day / 31.0)

    # 类别编码
    encoders = build_encoders(df, [c for c in CATEGORICALS if c in df.columns])
    X_cat, cat_names = ohe_transform(df, encoders)

    # 数值列：去掉 id 和已被OHE的列；同时去掉被替换的原始列（balance/duration/day）
    drop_cols = set(encoders.keys()) | {'id'}
    raw_drop = set()
    if 'balance' in df.columns: raw_drop.add('balance')
    if 'duration' in df.columns: raw_drop.add('duration')
    if 'day' in df.columns: raw_drop.add('day')

    num_cols = [c for c in df.columns if c not in drop_cols and c not in raw_drop]
    X_num = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)

    X = np.hstack([X_cat, X_num]) if X_cat.size else X_num
    feature_names = cat_names + num_cols
    meta = {
        'encoders': encoders,
        'numeric_cols': num_cols,        # 推理时重建数值（包含衍生名）
        'feature_names_all': feature_names  # 目前仅 OHE+数值/衍生
    }
    return X, meta

def compute_te_maps(df, y, fields, m=TE_M):
    maps = {}
    mu = float(np.mean(y))
    for f in fields:
        if f not in df.columns:
            continue
        s = df[f].astype(str).fillna('')
        stats = pd.DataFrame({'cat': s, 'y': y}).groupby('cat')['y'].agg(['sum', 'count']).reset_index()
        smoothed = (stats['sum'] + m * mu) / (stats['count'] + m)
        te_map = dict(zip(stats['cat'], smoothed))
        maps[f] = {'global_mean': mu, 'm': m, 'map': te_map}
    return maps

def apply_te(df, te_maps):
    mats, names = [], []
    for f, info in te_maps.items():
        if f not in df.columns:
            # 若字段不存在，跳过但保持一致性
            continue
        s = df[f].astype(str).fillna('')
        mu = info['global_mean']
        te_map = info['map']
        v = s.map(lambda x: te_map.get(x, mu)).astype(float).values.reshape(-1, 1)
        mats.append(v)
        names.append(f'te_{f}')
    if mats:
        te_mat = np.hstack(mats)
    else:
        te_mat = np.empty((len(df), 0), dtype=float)
    return te_mat, names

def kfold_rmse(Xz, y, l2, k=5, seed=42):
    n = Xz.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    rmses = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_tr, y_tr = Xz[tr_idx], y[tr_idx]
        X_va, y_va = Xz[val_idx], y[val_idx]

        m = Model()
        m.fit_closed_form(X_tr, y_tr, l2=l2)
        y_pred = m.predict(X_va)
        rmse = np.sqrt(np.mean((y_pred - y_va) ** 2))
        rmses.append(rmse)
    return float(np.mean(rmses))

def main():
    df = pd.read_csv(DATA_PATH)
    y = df['age'].astype(float).values
    df_feat = df.drop(columns=['age'])

    # 构建基础特征（OHE + 数值衍生）
    X_base, meta = build_features(df_feat)

    # 目标均值编码特征
    te_maps = compute_te_maps(df_feat, y, TE_FIELDS, m=TE_M)
    TE_mat, TE_names = apply_te(df_feat, te_maps)

    # 拼接 TE 到特征末尾
    X = np.hstack([X_base, TE_mat]) if TE_mat.size else X_base
    feature_names_all = meta['feature_names_all'] + TE_names

    # 标准化（全量）
    X_mean_all = X.mean(axis=0)
    X_std_all = X.std(axis=0)
    X_std_all[X_std_all == 0] = 1.0
    Xz_all = (X - X_mean_all) / X_std_all

    # 第一阶段：初训 + 解析型 SHAP
    model0 = Model()
    model0.fit_closed_form(Xz_all, y, l2=1e-2)
    w0 = model0.weights
    shap_abs = np.mean(np.abs(Xz_all * w0), axis=0)

    # 基于SHAP筛选：95%累计贡献，至少32维
    order = np.argsort(-shap_abs)
    cum = np.cumsum(shap_abs[order])
    total = cum[-1] if cum[-1] > 0 else 1.0
    k95 = np.searchsorted(cum, total * 0.95) + 1
    keep_k = max(32, int(k95))
    keep_idx_sorted = sorted(order[:keep_k])

    # 选中特征并重新标准化
    X_sel = X[:, keep_idx_sorted]
    X_mean = X_sel.mean(axis=0)
    X_std = X_sel.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xz = (X_sel - X_mean) / X_std

    # 第二阶段：5折CV选 L2
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]
    cv_scores = [(l2, kfold_rmse(Xz, y, l2=l2, k=5, seed=42)) for l2 in grid]
    best_l2, _ = min(cv_scores, key=lambda t: t[1])

    # 最终训练
    model = Model()
    model.fit_closed_form(Xz, y, l2=best_l2)

    # 年龄范围用于推理裁剪
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # 保存
    feature_names_sel = [feature_names_all[i] for i in keep_idx_sorted]
    save_obj = {
        'weights': model.weights,
        'bias': model.bias,
        'X_mean': X_mean,
        'X_std': X_std,
        'encoders': meta['encoders'],
        'numeric_cols': meta['numeric_cols'],   # 数值/衍生列表
        'feature_names_all': feature_names_all, # 全量名（含TE）
        'feature_names': feature_names_sel,     # 被选中的特征名
        'keep_idx': keep_idx_sorted,
        'best_l2': best_l2,
        'te_maps': te_maps,                     # 目标均值编码映射
        'te_names': TE_names,                   # TE 特征名顺序
        'y_min': y_min,
        'y_max': y_max
    }
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(save_obj, f)

    print(f'[OK] Trained with feature selection: kept {len(keep_idx_sorted)}/{X.shape[1]} features, best_l2={best_l2}')
    print(f'[OK] Saved -> {OUT_PATH}')

if __name__ == '__main__':
    main()