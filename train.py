import pandas as pd
import numpy as np
import pickle
from model import Model
from itertools import combinations

DATA_PATH = 'train.csv'
OUT_PATH = 'model_params.pkl'
CATEGORICALS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
TE_FIELDS = ['job', 'marital', 'education', 'month', 'contact', 'poutcome', 'housing']
TE_M = 30  # 增加平滑系数,减少过拟合

# ============ 扩展TE交互对 ============
TE_INTER_PAIRS = [
    ('te_marital', 'te_job'),
    ('te_job', 'te_education'),
    ('te_marital', 'te_education'),
    ('te_job', 'te_contact'),
    ('te_education', 'te_poutcome'),
    ('te_job', 'te_month'),          # 新增
    ('te_education', 'te_contact'),  # 新增
    ('te_marital', 'te_housing'),    # 新增
    ('te_poutcome', 'te_contact'),   # 新增
    ('te_job', 'te_poutcome'),       # 新增
]

def build_encoders(df, cats):
    encoders = {}
    for c in cats:
        vals = df[c].astype(str).fillna('')
        categories = list(pd.Index(vals).unique())
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

def signed_log1p_series(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors='coerce').fillna(0.0).astype(float)
    return (np.sign(v) * np.log1p(np.abs(v))).values

def build_features(df):
    df = df.copy()
    winsor = {}

    # ============ 新增领域知识特征 ============
    # 1. 职业-教育匹配度
    if 'job' in df.columns and 'education' in df.columns:
        high_edu_jobs = ['management', 'technician', 'admin.']
        df['job_edu_match'] = ((df['job'].isin(high_edu_jobs)) & 
                                (df['education'] == 'tertiary')).astype(int)
    
    # 2. 金融稳定性指标
    if 'balance' in df.columns:
        df['has_negative_balance'] = (df['balance'] < 0).astype(int)
    
    if 'loan' in df.columns and 'housing' in df.columns:
        df['has_any_debt'] = ((df['loan'] == 'yes') | (df['housing'] == 'yes')).astype(int)
        df['total_loans'] = (df['loan'] == 'yes').astype(int) + (df['housing'] == 'yes').astype(int)
    
    # 3. 联系频次特征
    if 'campaign' in df.columns and 'previous' in df.columns:
        df['total_contacts'] = df['campaign'] + df['previous']
        df['contact_ratio'] = df['campaign'] / (df['total_contacts'] + 1)
    
    if 'pdays' in df.columns:
        df['prev_contacted'] = (df['pdays'] >= 0).astype(int)
        pdays_safe = df['pdays'].clip(lower=1)
        if 'campaign' in df.columns:
            df['contact_intensity'] = df['campaign'] / (pdays_safe + 1)
    
    # 4. 年龄相关推断特征
    if 'marital' in df.columns:
        df['is_single'] = (df['marital'] == 'single').astype(int)
        df['is_divorced'] = (df['marital'] == 'divorced').astype(int)
    
    if 'poutcome' in df.columns:
        df['prev_success'] = (df['poutcome'] == 'success').astype(int)
    
    # ============ Winsorization优化(0.5%/99.5%) ============
    if 'balance' in df.columns:
        ql, qh = float(df['balance'].quantile(0.005)), float(df['balance'].quantile(0.995))
        winsor['balance'] = (ql, qh)
        bal_clip = df['balance'].clip(lower=ql, upper=qh)
        df['balance_log'] = signed_log1p_series(bal_clip)
        df['balance_log2'] = df['balance_log'] ** 2
        df['balance_log3'] = df['balance_log'] ** 3  # 新增三次方
        df['balance_sqrt'] = np.sign(bal_clip) * np.sqrt(np.abs(bal_clip))
    
    if 'duration' in df.columns:
        ql, qh = float(df['duration'].quantile(0.005)), float(df['duration'].quantile(0.995))
        winsor['duration'] = (ql, qh)
        dur_clip = df['duration'].clip(lower=ql, upper=qh)
        df['duration_log'] = signed_log1p_series(dur_clip)
        df['duration_log2'] = df['duration_log'] ** 2
        df['duration_log3'] = df['duration_log'] ** 3  # 新增三次方
        df['duration_sqrt'] = np.sqrt(dur_clip.clip(lower=0))
        df['duration_reciprocal'] = 1.0 / (df['duration_log'].abs() + 0.1)  # 新增倒数
    
    # ============ 增强交互特征 ============
    if 'balance_log' in df.columns and 'duration_log' in df.columns:
        df['balance_duration_inter'] = df['balance_log'] * df['duration_log']
        df['balance_duration_ratio'] = df['balance_log'] / (df['duration_log'].abs() + 0.1)  # 新增比值
    
    if 'balance' in df.columns and 'has_any_debt' in df.columns:
        df['balance_debt_ratio'] = bal_clip / (df['has_any_debt'] + 1)
    
    if 'campaign' in df.columns and 'duration_log' in df.columns:
        df['camp_dur_product'] = df['campaign'] * df['duration_log']  # 新增
    
    if 'balance' in df.columns and 'campaign' in df.columns:
        df['balance_per_contact'] = bal_clip / (df['campaign'] + 1)  # 新增
    
    # ============ 月份季节性特征 ============
    if 'month' in df.columns:
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        df['month_num'] = df['month'].map(month_map).fillna(0)
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        df['is_summer'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)
        df['is_year_end'] = df['month'].isin(['nov', 'dec']).astype(int)
        df['is_q1'] = df['month'].isin(['jan', 'feb', 'mar']).astype(int)
    
    # ============ 周期特征(天) ============
    if 'day' in df.columns:
        day = pd.to_numeric(df['day'], errors='coerce').fillna(0.0).astype(float)
        df['day_sin'] = np.sin(2.0 * np.pi * day / 31.0)
        df['day_cos'] = np.cos(2.0 * np.pi * day / 31.0)
        df['day_sc'] = df['day_sin'] * df['day_cos']
        df['is_month_start'] = (day <= 5).astype(int)
        df['is_month_end'] = (day >= 26).astype(int)
    
    # ============ Log1p变换 ============
    if 'campaign' in df.columns:
        df['campaign_log1p'] = np.log1p(pd.to_numeric(df['campaign'], errors='coerce').fillna(0.0).clip(lower=0.0))
    if 'previous' in df.columns:
        df['previous_log1p'] = np.log1p(pd.to_numeric(df['previous'], errors='coerce').fillna(0.0).clip(lower=0.0))
    if 'pdays' in df.columns:
        pdays_pos = pd.to_numeric(df['pdays'], errors='coerce').fillna(-1.0)
        pdays_pos = pdays_pos.where(pdays_pos > 0, 0.0)
        df['pdays_pos_log1p'] = np.log1p(pdays_pos)

    # 类别编码
    encoders = build_encoders(df, [c for c in CATEGORICALS if c in df.columns])
    X_cat, cat_names = ohe_transform(df, encoders)

    # 数值列
    drop_cols = set(encoders.keys()) | {'id'}
    raw_drop = {'balance', 'duration', 'day', 'month_num'}  # 排除原始列和中间变量
    num_cols = [c for c in df.columns if c not in drop_cols and c not in raw_drop]
    X_num = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)

    X = np.hstack([X_cat, X_num]) if X_cat.size else X_num
    feature_names = cat_names + num_cols
    meta = {
        'encoders': encoders,
        'numeric_cols': num_cols,
        'feature_names_all': feature_names,
        'winsor': winsor
    }
    return X, meta

# ============ K折Target Encoding ============
def compute_te_maps_kfold(df, y, fields, m=TE_M, n_folds=5):
    """使用K折避免TE过拟合"""
    maps = {}
    mu = float(np.mean(y))
    
    # 手动实现K折
    n = len(df)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    for f in fields:
        if f not in df.columns:
            continue
        s = df[f].astype(str).fillna('')
        te_values = np.zeros(n)
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            # 仅用训练折计算编码
            train_stats = pd.DataFrame({
                'cat': s.iloc[train_idx],
                'y': y[train_idx]
            }).groupby('cat')['y'].agg(['sum', 'count'])
            
            smoothed_map = {}
            for cat in train_stats.index:
                smoothed_map[cat] = (train_stats.loc[cat, 'sum'] + m * mu) / \
                                   (train_stats.loc[cat, 'count'] + m)
            
            # 应用到验证折
            te_values[val_idx] = s.iloc[val_idx].map(lambda x: smoothed_map.get(x, mu)).values
        
        # 最终用全量数据计算映射(用于推理)
        stats = pd.DataFrame({'cat': s, 'y': y}).groupby('cat')['y'].agg(['sum', 'count'])
        smoothed = (stats['sum'] + m * mu) / (stats['count'] + m)
        te_map = dict(zip(stats.index, smoothed))
        
        maps[f] = {'global_mean': mu, 'm': m, 'map': te_map}
    
    return maps

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
    for a, b in pairs:
        if a in name2idx and b in name2idx:
            va = TE_mat[:, name2idx[a]].reshape(-1, 1)
            vb = TE_mat[:, name2idx[b]].reshape(-1, 1)
            mats.append(va * vb)
            names.append(f'{a}*{b}')
    if mats:
        M = np.hstack(mats)
    else:
        M = np.empty((TE_mat.shape[0], 0), dtype=float)
    return M, names

# ============ 多项式特征 ============
def add_polynomial_features(X, feature_names, top_k=10):
    """仅对重要特征添加二次项"""
    poly_features = []
    poly_names = []
    
    # 选择前top_k个特征
    n_features = min(top_k, X.shape[1])
    
    for i, j in combinations(range(n_features), 2):
        poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        poly_names.append(f'poly_{feature_names[i]}*{feature_names[j]}')
    
    if poly_features:
        return np.hstack([X] + poly_features), feature_names + poly_names
    return X, feature_names

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

    print("=" * 60)
    print("🚀 开始特征工程...")
    print("=" * 60)
    
    # 基础特征
    X_base, meta = build_features(df_feat)
    print(f"✓ 基础特征: {X_base.shape[1]}维")

    # K折Target Encoding
    te_maps = compute_te_maps_kfold(df_feat, y, TE_FIELDS, m=TE_M, n_folds=5)
    TE_mat, TE_names = apply_te(df_feat, te_maps)
    print(f"✓ TE特征: {len(TE_names)}维")

    # TE交互
    TE_inter_mat, TE_inter_names = build_te_interactions(TE_mat, TE_names, TE_INTER_PAIRS)
    print(f"✓ TE交互: {len(TE_inter_names)}对")

    # 拼接特征
    X = X_base
    feature_names_all = meta['feature_names_all']
    if TE_mat.size:
        X = np.hstack([X, TE_mat])
        feature_names_all = feature_names_all + TE_names
    if TE_inter_mat.size:
        X = np.hstack([X, TE_inter_mat])
        feature_names_all = feature_names_all + TE_inter_names
    
    print(f"✓ 全量特征: {X.shape[1]}维")

    # 标准化(全量)
    X_mean_all = X.mean(axis=0)
    X_std_all = X.std(axis=0)
    X_std_all[X_std_all == 0] = 1.0
    Xz_all = (X - X_mean_all) / X_std_all

    print("\n" + "=" * 60)
    print("📊 SHAP特征选择...")
    print("=" * 60)
    
    # 第一阶段:初训+SHAP
    model0 = Model()
    model0.fit_closed_form(Xz_all, y, l2=1e-2)
    w0 = model0.weights
    shap_abs = np.mean(np.abs(Xz_all * w0), axis=0)

    # ============ 递增式特征选择 ============
    order = np.argsort(-shap_abs)
    best_k = 28
    best_score = float('inf')

    print("开始递增式特征选择 (28→120, 步长=4)...")
    for k in range(28, min(120, len(order)), 4):
        keep_idx_temp = sorted(order[:k])
        X_temp = X[:, keep_idx_temp]
        
        # 标准化
        X_mean_temp = X_temp.mean(axis=0)
        X_std_temp = X_temp.std(axis=0)
        X_std_temp[X_std_temp == 0] = 1.0
        Xz_temp = (X_temp - X_mean_temp) / X_std_temp
        
        # 快速3折验证
        score = kfold_rmse(Xz_temp, y, l2=1e-2, k=3, seed=42)
        
        if score < best_score:
            best_score = score
            best_k = k
            print(f"  ✓ k={k:3d}, CV-RMSE={score:.4f} (更新最优)")
        elif score > best_score * 1.005:  # 连续恶化超5‰则停止
            print(f"  ✗ k={k:3d}, CV-RMSE={score:.4f} (早停)")
            break

    print(f"\n最优特征数: {best_k}, CV-RMSE: {best_score:.4f}")
    keep_idx_sorted = sorted(order[:best_k])

    # 选中特征并标准化
    X_sel = X[:, keep_idx_sorted]
    
    # ============ 添加多项式特征 ============
    feature_names_sel = [feature_names_all[i] for i in keep_idx_sorted]
    X_sel, feature_names_sel = add_polynomial_features(X_sel, feature_names_sel, top_k=10)
    print(f"✓ 多项式扩展: {X_sel.shape[1]}维 (含Top-10的{len(list(combinations(range(10), 2)))}个二次项)")
    
    X_mean = X_sel.mean(axis=0)
    X_std = X_sel.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xz = (X_sel - X_mean) / X_std

    print("\n" + "=" * 60)
    print("🔧 超参数搜索...")
    print("=" * 60)
    
    # ============ 三阶段超参搜索 ============
    # 粗搜
    print("\n[阶段1] 粗搜 (1e-5 → 100, 对数间隔)...")
    coarse_grid = np.logspace(-5, 2, 15)
    coarse_scores = [(l2, kfold_rmse(Xz, y, l2, k=3, seed=42)) for l2 in coarse_grid]
    best_coarse = min(coarse_scores, key=lambda t: t[1])[0]
    best_coarse_score = min(coarse_scores, key=lambda t: t[1])[1]
    print(f"粗搜最优: λ={best_coarse:.2e}, RMSE={best_coarse_score:.4f}")
    
    # 细搜
    print(f"\n[阶段2] 细搜 (λ={best_coarse*0.05:.2e} → {best_coarse*20:.2e}, 30点)...")
    fine_grid = np.linspace(best_coarse * 0.05, best_coarse * 20, 30)
    fine_scores = [(l2, kfold_rmse(Xz, y, l2, k=5, seed=42)) for l2 in fine_grid]
    best_fine = min(fine_scores, key=lambda t: t[1])[0]
    best_fine_score = min(fine_scores, key=lambda t: t[1])[1]
    print(f"细搜最优: λ={best_fine:.2e}, RMSE={best_fine_score:.4f}")
    
    # 超细搜
    print(f"\n[阶段3] 超细搜 (λ={best_fine*0.5:.2e} → {best_fine*1.5:.2e}, 20点)...")
    ultra_fine_grid = np.linspace(best_fine * 0.5, best_fine * 1.5, 20)
    ultra_scores = [(l2, kfold_rmse(Xz, y, l2, k=5, seed=42)) for l2 in ultra_fine_grid]
    best_l2, best_rmse = min(ultra_scores, key=lambda t: t[1])
    print(f"最终最优: λ={best_l2:.2e}, RMSE={best_rmse:.4f}")

    print("\n" + "=" * 60)
    print("🎯 Bootstrap集成训练...")
    print("=" * 60)
    
    # ============ Bootstrap聚合 (3个模型) ============
    n_models = 3
    models = []
    np.random.seed(42)

    for i in range(n_models):
        boot_idx = np.random.choice(len(Xz), size=len(Xz), replace=True)
        Xz_boot = Xz[boot_idx]
        y_boot = y[boot_idx]
        
        m = Model()
        m.fit_closed_form(Xz_boot, y_boot, l2=best_l2)
        models.append(m)
        print(f"  ✓ 模型 {i+1}/{n_models} 训练完成")

    # 平均权重
    avg_weights = np.mean([m.weights for m in models], axis=0)
    avg_bias = np.mean([m.bias for m in models])

    model = Model()
    model.weights = avg_weights
    model.bias = avg_bias
    print(f"✓ 集成完成 (权重平均)")

    # 年龄范围(放宽10%)
    y_min_raw, y_max_raw = float(np.min(y)), float(np.max(y))
    margin = (y_max_raw - y_min_raw) * 0.1
    y_min = y_min_raw - margin
    y_max = y_max_raw + margin

    # 保存
    save_obj = {
        'weights': model.weights,
        'bias': model.bias,
        'X_mean': X_mean,
        'X_std': X_std,
        'encoders': meta['encoders'],
        'numeric_cols': meta['numeric_cols'],
        'feature_names_all': feature_names_all,
        'feature_names': feature_names_sel,
        'keep_idx': keep_idx_sorted,
        'best_l2': best_l2,
        'te_maps': te_maps,
        'te_names': TE_names,
        'te_inter_pairs': TE_INTER_PAIRS,
        'winsor': meta.get('winsor', {}),
        'y_min': y_min,
        'y_max': y_max,
        'poly_top_k': 10,
        'n_bootstrap': n_models
    }
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(save_obj, f)

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"📊 特征维度: {len(keep_idx_sorted)} → {X_sel.shape[1]} (含多项式)")
    print(f"🔧 最优L2: {best_l2:.6f}")
    print(f"🎯 CV-RMSE: {best_rmse:.4f}")
    print(f"📦 模型集成: {n_models}个Bootstrap模型")
    print(f"💾 已保存至: {OUT_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    main()