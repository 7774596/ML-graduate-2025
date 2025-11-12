import os
import argparse
import numpy as np
from typing import List, Tuple
from model import Model


def parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(np.float32, copy=False)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    mean = X.mean(axis=0, dtype=np.float32)
    std = X.std(axis=0, dtype=np.float32)
    std[std < 1e-6] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def median_heuristic_gamma(Xs: np.ndarray, seed: int = 42, max_samples: int = 800) -> float:
    n = Xs.shape[0]
    m = min(n, max_samples)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, m, replace=False)
    X = Xs[idx]
    Xn = np.sum(X * X, axis=1, keepdims=True)
    D = Xn + Xn.T - 2.0 * (X @ X.T)
    iu = np.triu_indices_from(D, k=1)
    dist = D[iu]
    med = np.median(dist)
    if med <= 0:
        med = np.mean(dist[dist > 0]) if np.any(dist > 0) else 1.0
    return 1.0 / (2.0 * float(med))


def accuracy(model: Model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float(np.mean(preds == y.ravel()))


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter tuning for Kernel SVM (grid search)")
    p.add_argument('--kernel-list', type=str, default=os.getenv('TUNE_KERNELS', 'rbf'),
                   help='Comma-separated kernels to try: rbf,poly')
    p.add_argument('--c-list', type=str, default=os.getenv('TUNE_C_LIST', '2,5,8,10'))
    p.add_argument('--gamma-scales', type=str, default=os.getenv('TUNE_GAMMA_SCALES', '0.5,1,2'),
                   help='Scales applied to auto gamma for rbf')
    p.add_argument('--degrees', type=str, default=os.getenv('TUNE_DEGREES', '3'),
                   help='Degrees for poly kernel, comma-separated')
    p.add_argument('--coef0-list', type=str, default=os.getenv('TUNE_COEF0_LIST', '1.0'),
                   help='coef0 list for poly kernel, comma-separated')
    p.add_argument('--val-ratio', type=float, default=float(os.getenv('TUNE_VAL_RATIO', '0.2')))
    p.add_argument('--tune-subsample', type=int, default=int(os.getenv('TUNE_SUBSAMPLE', '0')),
                   help='If >0, subsample this many examples for tuning (faster)')
    p.add_argument('--seed', type=int, default=int(os.getenv('TUNE_SEED', '42')))
    p.add_argument('--max-passes', type=int, default=int(os.getenv('TUNE_MAX_PASSES', '4')))
    p.add_argument('--max-iter', type=int, default=int(os.getenv('TUNE_MAX_ITER', '1200')))
    p.add_argument('--final-max-passes', type=int, default=int(os.getenv('FINAL_MAX_PASSES', '5')))
    p.add_argument('--final-max-iter', type=int, default=int(os.getenv('FINAL_MAX_ITER', '1500')))
    p.add_argument('--output', type=str, default=os.getenv('MODEL_PATH', 'svm_model.npz'))
    p.add_argument('--log', type=str, default=os.getenv('TUNE_LOG', 'tune_results.csv'))
    p.add_argument('--verbose', action='store_true', default=os.getenv('SVM_VERBOSE', '1') != '0')
    return p.parse_args()


def main():
    args = parse_args()
    kernels = [k.strip() for k in args.kernel_list.split(',') if k.strip()]
    c_list = parse_list_float(args.c_list)
    gamma_scales = parse_list_float(args.gamma_scales)
    degrees = parse_list_int(args.degrees)
    coef0_list = parse_list_float(args.coef0_list)

    # Load data
    data = np.load('data/train.npz')
    X = data['X_train']
    y = data['y_train'].astype(np.int32)

    n = X.shape[0]
    rng = np.random.RandomState(args.seed)

    # Optional subsample for faster tuning
    if args.tune_subsample and args.tune_subsample < n:
        idx = rng.choice(n, args.tune_subsample, replace=False)
        X = X[idx]
        y = y[idx]
        n = X.shape[0]
        print(f"[Tune] Subsampled to {n} examples for tuning.")

    # Train/val split
    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]
    split = int((1.0 - args.val_ratio) * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # For RBF gamma auto heuristic we compute on standardized X_tr
    X_tr_std, mean_tr, std_tr = standardize(X_tr)
    auto_gamma = median_heuristic_gamma(X_tr_std, seed=args.seed)
    print(f"[Tune] Auto gamma (median heuristic on train split) = {auto_gamma:.6f}")

    # Prepare log
    if not os.path.exists(args.log):
        with open(args.log, 'w', encoding='utf-8') as f:
            f.write('kernel,C,gamma,degree,coef0,acc\n')

    best = {
        'acc': -1.0,
        'kernel': None,
        'C': None,
        'gamma': None,
        'degree': None,
        'coef0': None,
    }

    # Grid search
    for kernel in kernels:
        if kernel == 'rbf':
            for C in c_list:
                for scale in gamma_scales:
                    gamma = auto_gamma * scale
                    model = Model(
                        n_features=28*28,
                        n_classes=10,
                        kernel='rbf',
                        C=C,
                        gamma=gamma,
                        verbose=args.verbose,
                        max_passes=args.max_passes,
                        max_iter=args.max_iter,
                        seed=args.seed,
                    )
                    print(f"[Tune] Train kernel=rbf C={C} gamma={gamma:.6f}")
                    model.fit(X_tr, y_tr)
                    acc = accuracy(model, X_val, y_val)
                    print(f"[Tune] Val acc={acc:.4f}")
                    with open(args.log, 'a', encoding='utf-8') as f:
                        f.write(f"rbf,{C},{gamma:.8f},,,{acc:.6f}\n")
                    if acc > best['acc']:
                        best.update({'acc': acc, 'kernel': 'rbf', 'C': C, 'gamma': gamma, 'degree': None, 'coef0': None})
        elif kernel == 'poly':
            for C in c_list:
                for degree in degrees:
                    for coef0 in coef0_list:
                        # gamma for poly: use 1/n_features or scaled auto? We'll use 1/n_features as simple default
                        gamma_poly = 1.0 / float(28*28)
                        model = Model(
                            n_features=28*28,
                            n_classes=10,
                            kernel='poly',
                            C=C,
                            gamma=gamma_poly,
                            degree=degree,
                            coef0=coef0,
                            verbose=args.verbose,
                            max_passes=args.max_passes,
                            max_iter=args.max_iter,
                            seed=args.seed,
                        )
                        print(f"[Tune] Train kernel=poly C={C} degree={degree} coef0={coef0}")
                        model.fit(X_tr, y_tr)
                        acc = accuracy(model, X_val, y_val)
                        print(f"[Tune] Val acc={acc:.4f}")
                        with open(args.log, 'a', encoding='utf-8') as f:
                            f.write(f"poly,{C},{gamma_poly:.8f},{degree},{coef0},{acc:.6f}\n")
                        if acc > best['acc']:
                            best.update({'acc': acc, 'kernel': 'poly', 'C': C, 'gamma': gamma_poly, 'degree': degree, 'coef0': coef0})

    print("[Tune] Best config:")
    print(best)

    # Retrain on full dataset with best params (no subsample) and stronger convergence
    data_full = np.load('data/train.npz')
    X_full = data_full['X_train']
    y_full = data_full['y_train']

    if best['kernel'] == 'rbf':
        model_best = Model(
            n_features=28*28,
            n_classes=10,
            kernel='rbf',
            C=best['C'],
            gamma=best['gamma'],
            verbose=True,
            max_passes=args.final_max_passes,
            max_iter=args.final_max_iter,
            seed=args.seed,
        )
    else:
        model_best = Model(
            n_features=28*28,
            n_classes=10,
            kernel='poly',
            C=best['C'],
            gamma=best['gamma'],
            degree=best['degree'],
            coef0=best['coef0'],
            verbose=True,
            max_passes=args.final_max_passes,
            max_iter=args.final_max_iter,
            seed=args.seed,
        )

    print("[Tune] Retraining best model on full dataset...")
    model_best.fit(X_full, y_full)
    model_best.save(args.output)
    print(f"[Tune] Saved best model to {args.output}")


if __name__ == '__main__':
    main()
