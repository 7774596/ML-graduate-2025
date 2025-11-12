import os
import argparse
import numpy as np
from model import Model


def parse_args():
    p = argparse.ArgumentParser(description="Train Kernel SVM and save to npz")
    p.add_argument('--kernel', type=str, default=os.getenv('SVM_KERNEL', 'rbf'), choices=['rbf', 'poly'])
    p.add_argument('--C', type=float, default=float(os.getenv('SVM_C', '2.0')))
    p.add_argument('--gamma', type=float, default=float(os.getenv('SVM_GAMMA')) if os.getenv('SVM_GAMMA') else None)
    p.add_argument('--degree', type=int, default=int(os.getenv('SVM_DEGREE', '3')))
    p.add_argument('--coef0', type=float, default=float(os.getenv('SVM_COEF0', '1.0')))
    p.add_argument('--verbose', action='store_true', default=os.getenv('SVM_VERBOSE', '1') != '0')
    p.add_argument('--subsample', type=int, default=int(os.getenv('SVM_SUBSAMPLE')) if os.getenv('SVM_SUBSAMPLE') else None)
    p.add_argument('--max_passes', type=int, default=int(os.getenv('SVM_MAX_PASSES', '3')))
    p.add_argument('--max_iter', type=int, default=int(os.getenv('SVM_MAX_ITER', '800')))
    p.add_argument('--output', type=str, default='svm_model.npz')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        data = np.load('data/train.npz')
        X_train = data['X_train']
        y_train = data['y_train']
    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {e}")

    if args.subsample is not None and args.subsample < X_train.shape[0]:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], args.subsample, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        if args.verbose:
            print(f"Subsampled training set to {args.subsample} examples for debug.")

    model = Model(
        n_features=28 * 28,
        n_classes=10,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        verbose=args.verbose,
        max_passes=args.max_passes,
        max_iter=args.max_iter,
        seed=42,
    )

    print(f"Training Kernel SVM (kernel={args.kernel}) ...")
    model.fit(X_train, y_train)
    print("Training finished. Saving model...")
    model.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
