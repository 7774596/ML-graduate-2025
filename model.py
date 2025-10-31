import numpy as np
import pickle
import os

class Model:
    def __init__(self):
        """
        初始化线性回归模型
        """
        self.weights = None
        self.bias = None

    def predict(self, X):
        """
        预测样本的值

        Args:
            X: 输入特征，形状为 (n_samples, n_features) 的numpy数组

        Returns:
            numpy.ndarray: 形状为 (n_samples,) 的预测值数组
        """
        return np.dot(X, self.weights) + self.bias

    def fit_closed_form(self, X, y, l2=1e-3):
        """
        使用闭式解（带L2正则）拟合线性回归

        Args:
            X: numpy.ndarray, shape (n_samples, n_features)
            y: numpy.ndarray, shape (n_samples,)
            l2: float, L2正则系数
        """
        n_features = X.shape[1]
        # 增加偏置列到X以同时学习bias
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  # shape (n, d+1)
        A = X_aug.T.dot(X_aug)
        # 对角上对bias不正则化
        A[-1, -1] += 0.0
        A += l2 * np.eye(n_features + 1)
        b = X_aug.T.dot(y)
        params = np.linalg.solve(A, b)  # shape (d+1,)
        self.weights = params[:-1]
        self.bias = params[-1]

    def save_params(self, path: str):
        """
        保存模型权重到文件（pickle）
        """
        obj = {'weights': self.weights, 'bias': self.bias}
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load_params(self, path: str):
        """
        从文件加载模型权重
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.weights = obj['weights']
        self.bias = obj['bias']