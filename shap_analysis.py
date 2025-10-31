# ...existing code...
import pickle
import pandas as pd
import numpy as np
from train import build_features

def apply_te(df, te_maps):
    mats, names = [], []
    for f, info in te_maps.items():
        if f not in df.columns:
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

def build_te_interactions(TE_mat: np.ndarray, TE_names, pairs):
    if TE_mat.size == 0:
        return np.empty((TE_mat.shape[0], 0)), []
    name2idx = {n: i for i, n in enumerate(TE_names)}
    mats, names = [], []
    for a_name, b_name in pairs:  # 避免覆盖 bias 变量
        if a_name in name2idx and b_name in name2idx:
            va = TE_mat[:, name2idx[a_name]].reshape(-1, 1)
            vb = TE_mat[:, name2idx[b_name]].reshape(-1, 1)
            mats.append(va * vb)
            names.append(f'{a_name}*{b_name}')
    if mats:
        M = np.hstack(mats)
    else:
        M = np.empty((TE_mat.shape[0], 0), dtype=float)
    return M, names

def main():
    with open('model_params.pkl', 'rb') as f:
        obj = pickle.load(f)
    w = obj['weights']; b = obj['bias']
    X_mean = obj['X_mean']; X_std = obj['X_std']
    feat_all = obj['feature_names_all']
    feat_sel = obj['feature_names']
    keep_idx = obj.get('keep_idx', None)
    te_maps = obj.get('te_maps', {})
    te_inter_pairs = obj.get('te_inter_pairs', [])

    # 基础特征
    df = pd.read_csv('train.csv')
    X_base, meta = build_features(df.drop(columns=['age']))

    # TE + TE交互
    TE_mat, TE_names = apply_te(df.drop(columns=['age']), te_maps)
    TE_inter_mat, TE_inter_names = build_te_interactions(TE_mat, TE_names, te_inter_pairs)

    # 全量
    X_full = X_base
    names_full = meta['feature_names_all']
    if TE_mat.size:
        X_full = np.hstack([X_full, TE_mat])
        names_full = names_full + TE_names
    if TE_inter_mat.size:
        X_full = np.hstack([X_full, TE_inter_mat])
        names_full = names_full + TE_inter_names

    assert names_full == feat_all, "特征列顺序不一致，请重新训练。"

    # 选中特征并标准化
    X = X_full[:, keep_idx] if keep_idx is not None else X_full
    Z = (X - X_mean) / X_std

    shap_vals = Z * w
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(-mean_abs)

    print("\n[Top-20 特征重要性 - 按单特征]")
    names = feat_sel if feat_sel is not None else feat_all
    for idx in order[:20]:
        print(f"{names[idx]:<24s}  mean|SHAP|={mean_abs[idx]:.4f}")

    # 按字段聚合（交互贡献计入双方）
    group_imp = {}
    fields = []
    for name in names:
        field = name.split('=')[0] if '=' in name else name
        if '*' in field:
            a_name, b_name = field.split('*')
            fields.extend([a_name, b_name])
        else:
            fields.append(field)
    uniq_fields = []
    for f_name in fields:
        if f_name not in uniq_fields:
            uniq_fields.append(f_name)

    for f_name in uniq_fields:
        idxs = [i for i, nm in enumerate(names)
                if (nm.split('=')[0] == f_name) or ('*' in nm and f_name in nm.split('*'))]
        group_imp[f_name] = float(np.sum(mean_abs[idxs]))

    print("\n[按字段聚合的重要性]")
    for f_name, v_imp in sorted(group_imp.items(), key=lambda x: -x[1])[:20]:
        print(f"{f_name:<16s}  mean|SHAP|={v_imp:.4f}")

    print("\nbase_value(bias):", float(b))

if __name__ == '__main__':
    main()
# ...existing code...