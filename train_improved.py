import argparse
import time
import numpy as np
from model import Model


def load_data():
    data = np.load("data/train.npz")
    return data["X_train"], data["y_train"]


def augment_sample(img, max_shift=2, noise_scale=0.03):
    """
    对单张 28x28 图像做轻量增强：平移（roll）和加高斯噪声。
    保持像素范围在 [0,1]
    """
    # 随机平移
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    aug = np.roll(img, shift=dx, axis=0)
    aug = np.roll(aug, shift=dy, axis=1)

    # 添加小量高斯噪声
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=aug.shape)
    aug = aug + noise
    aug = np.clip(aug, 0.0, 1.0)
    return aug


def oversample_minority(X, y, target_per_class=1000, classes_to_boost=(8, 9)):
    """
    对指定类别进行过采样并做轻量增强，返回扩增后的 (X_new, y_new)
    """
    X_list = [x.copy() for x in X]
    y_list = [int(v) for v in y]

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    for cls in classes_to_boost:
        current = class_counts.get(cls, 0)
        if current >= target_per_class:
            continue

        need = target_per_class - current
        # 从该类中随机采样并增强
        indices = np.where(y == cls)[0]
        if len(indices) == 0:
            continue

        for _ in range(need):
            idx = np.random.choice(indices)
            img = X[idx]
            aug = augment_sample(img)
            X_list.append(aug)
            y_list.append(int(cls))

    X_new = np.stack(X_list, axis=0)
    y_new = np.array(y_list, dtype=np.int64)
    return X_new, y_new


def evaluate(model, X, y):
    t0 = time.time()
    y_pred = model.predict(X)
    t1 = time.time()
    pred_time = t1 - t0

    acc = np.mean(y_pred == y)
    n_classes = len(np.unique(y))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y, y_pred):
        cm[int(a), int(b)] += 1

    print(f"Accuracy: {acc:.4f}")
    print(f"Prediction time: {pred_time:.2f}s")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    return acc, pred_time, cm


def main(args):
    print("Loading data...")
    X, y = load_data()
    print(f"Original dataset: X={X.shape}, y={y.shape}")

    if args.quick:
        # 快速模式：使用小子集与少量迭代用于功能验证
        n_use = min(1200, len(X))
        X = X[:n_use]
        y = y[:n_use]
        print(f"Quick mode: using first {n_use} samples for a quick check")

    # 对类别8和9做过采样增强（只在非-quick模式下扩增较多）
    target = 1000 if not args.quick else 400
    X_aug, y_aug = oversample_minority(X, y, target_per_class=target, classes_to_boost=(8, 9))
    print(f"After augmentation: X={X_aug.shape}, y={y_aug.shape}")

    # 模型超参数（推荐的改进参数）
    C = 1.2
    gamma = 0.0006
    max_iter = 150 if not args.quick else 30

    model = Model(C=C, kernel='rbf', gamma=gamma, max_iter=max_iter, tol=1e-3, verbose=True)

    print(f"Training improved SVM: C={C}, gamma={gamma}, max_iter={max_iter}")
    t0 = time.time()
    model.fit(X_aug, y_aug)
    t1 = time.time()
    print(f"Training finished in {t1 - t0:.2f}s")

    # 评估并保存
    evaluate(model, X, y)
    out_path = args.out if args.out else 'svm_improved_model.npz'
    model.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick run for testing (small subset)')
    parser.add_argument('--out', type=str, default='svm_improved_model.npz', help='Output model path')
    args = parser.parse_args()
    main(args)
