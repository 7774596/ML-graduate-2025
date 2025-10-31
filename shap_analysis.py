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

def main():
    with open('model_params.pkl', 'rb') as f:
        obj = pickle.load(f)
    w = obj['weights']; b = obj['bias']
    X_mean = obj['X_mean']; X_std = obj['X_std']
    feat_all = obj['feature_names_all']
    feat_sel = obj['feature_names']
    keep_idx = obj.get('keep_idx', None)
    te_maps = obj.get('te_maps', {})

    # 基础特征
    df = pd.read_csv('train.csv')
    X_base, meta = build_features(df.drop(columns=['age']))

    # TE 特征
    TE_mat, TE_names = apply_te(df.drop(columns=['age']), te_maps)

    # 全量
    X_full = np.hstack([X_base, TE_mat]) if TE_mat.size else X_base
    assert meta['feature_names_all'] + TE_names == feat_all, "特征列顺序不一致，请重新训练。"

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

    # 按字段聚合
    group_imp = {}
    fields = []
    for name in names:
        field = name.split('=')[0] if '=' in name else name
        fields.append(field)
    uniq_fields = sorted(set(fields), key=fields.index)

    for f in uniq_fields:
        idxs = [i for i, fld in enumerate(fields) if fld == f]
        group_imp[f] = float(np.sum(mean_abs[idxs]))

    print("\n[按字段聚合的重要性]")
    for f, v in sorted(group_imp.items(), key=lambda x: -x[1])[:20]:
        print(f"{f:<16s}  mean|SHAP|={v:.4f}")

    print("\nbase_value(bias):", float(b))

if __name__ == '__main__':
    main()