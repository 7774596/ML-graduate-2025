import numpy as np


class SVMKernel:
    """核函数类"""
    
    @staticmethod
    def linear(X1, X2):
        """线性核: K(x1, x2) = x1^T * x2"""
        return np.dot(X1, X2.T)
    
    @staticmethod
    def polynomial(X1, X2, degree=3, coef0=1):
        """多项式核: K(x1, x2) = (gamma * x1^T * x2 + coef0)^degree"""
        return (np.dot(X1, X2.T) + coef0) ** degree
    
    @staticmethod
    def rbf(X1, X2, gamma=0.01):
        """
        RBF(高斯)核: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        使用矩阵运算优化: ||x1-x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1^T*x2
        """
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * K)


class BinarySVM:
    """
    二分类SVM实现 (One-vs-One策略)
    使用SMO (Sequential Minimal Optimization) 算法求解
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma=0.01, degree=3, coef0=1, 
                 max_iter=100, tol=1e-3):
        """
        参数:
            C: 正则化参数
            kernel: 核函数类型 ('linear', 'poly', 'rbf')
            gamma: RBF核的gamma参数
            degree: 多项式核的degree参数
            coef0: 多项式核的coef0参数
            max_iter: 最大迭代次数
            tol: 容忍度
        """
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        
        # 模型参数
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None
        
    def _kernel(self, X1, X2):
        """计算核函数"""
        if self.kernel_type == 'linear':
            return SVMKernel.linear(X1, X2)
        elif self.kernel_type == 'poly':
            return SVMKernel.polynomial(X1, X2, self.degree, self.coef0)
        elif self.kernel_type == 'rbf':
            return SVMKernel.rbf(X1, X2, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def fit(self, X, y):
        """
        训练SVM模型
        Args:
            X: 训练数据, shape (n_samples, n_features)
            y: 标签 (+1 或 -1), shape (n_samples,)
        """
        n_samples = X.shape[0]
        
        # 初始化alpha
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # 计算核矩阵
        K = self._kernel(X, X)
        
        # 简化版SMO算法
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            
            for i in range(n_samples):
                # 计算预测值
                prediction = np.sum(self.alpha * y * K[:, i]) + self.b
                
                # 计算误差
                E_i = prediction - y[i]
                
                # 检查KKT条件
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # 随机选择第二个alpha
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # 计算eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # 计算E_j
                    prediction_j = np.sum(self.alpha * y * K[:, j]) + self.b
                    E_j = prediction_j - y[j]
                    
                    # 保存旧的alpha值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # 更新alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    # 如果变化太小,跳过
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # 更新b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            # 检查收敛
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        
        # 提取支持向量
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.support_alpha = self.alpha[sv_indices]
    
    def predict(self, X):
        """
        预测
        Args:
            X: 测试数据, shape (n_samples, n_features)
        Returns:
            predictions: 预测标签 (+1 or -1)
        """
        if self.support_vectors is None:
            raise ValueError("Model not trained yet!")
        
        K = self._kernel(X, self.support_vectors)
        prediction = np.sum(self.support_alpha * self.support_labels * K, axis=1) + self.b
        return np.sign(prediction)


class Model:
    """
    多分类SVM (One-vs-One策略)
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma=0.01, degree=3, coef0=1,
                 max_iter=100, tol=1e-3, verbose=True):
        """
        参数:
            C: 正则化参数
            kernel: 核函数类型 ('linear', 'poly', 'rbf')
            gamma: RBF核的gamma参数
            degree: 多项式核的degree参数
            coef0: 多项式核的coef0参数
            max_iter: 最大迭代次数
            tol: 容忍度
            verbose: 是否打印训练信息
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.classifiers = {}
        self.classes = None
        self.X_mean = None
        self.X_std = None
        
    def _normalize(self, X, fit=False):
        """数据标准化"""
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std < 1e-6] = 1.0
        
        X_norm = (X - self.X_mean) / self.X_std
        return X_norm
    
    def fit(self, X, y):
        """
        训练多分类SVM (One-vs-One)
        Args:
            X: 训练数据, shape (n_samples, H, W) or (n_samples, n_features)
            y: 标签, shape (n_samples,)
        """
        # 标准化
        X = self._normalize(X, fit=True)
        y = np.asarray(y, dtype=np.int64).ravel()
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        if self.verbose:
            print(f"Training {n_classes}-class SVM with One-vs-One strategy...")
            print(f"Kernel: {self.kernel}, C: {self.C}")
            if self.kernel == 'rbf':
                print(f"Gamma: {self.gamma}")
            elif self.kernel == 'poly':
                print(f"Degree: {self.degree}, Coef0: {self.coef0}")
        
        # 训练所有的二分类器 (One-vs-One)
        total_pairs = n_classes * (n_classes - 1) // 2
        pair_count = 0
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                pair_count += 1
                
                # 选择两个类别的数据
                mask = (y == self.classes[i]) | (y == self.classes[j])
                X_pair = X[mask]
                y_pair = y[mask]
                
                # 转换标签为 +1/-1
                y_binary = np.where(y_pair == self.classes[i], 1, -1)
                
                # 训练二分类器
                clf = BinarySVM(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    max_iter=self.max_iter,
                    tol=self.tol
                )
                clf.fit(X_pair, y_binary)
                
                self.classifiers[(self.classes[i], self.classes[j])] = clf
                
                if self.verbose:
                    print(f"  Trained pair {pair_count}/{total_pairs}: "
                          f"class {self.classes[i]} vs class {self.classes[j]}")
        
        if self.verbose:
            print("Training completed!")
    
    def predict(self, X):
        """
        预测
        Args:
            X: 测试数据, shape (n_samples, H, W) or (n_samples, n_features)
        Returns:
            predictions: 预测标签, shape (n_samples,)
        """
        # 标准化
        X = self._normalize(X, fit=False)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # 投票矩阵
        votes = np.zeros((n_samples, n_classes), dtype=int)
        
        # 对每个分类器进行预测并投票
        for (i, j), clf in self.classifiers.items():
            predictions = clf.predict(X)
            
            # 投票
            i_idx = np.where(self.classes == i)[0][0]
            j_idx = np.where(self.classes == j)[0][0]
            
            votes[predictions == 1, i_idx] += 1
            votes[predictions == -1, j_idx] += 1
        
        # 选择得票最多的类别
        predicted_indices = np.argmax(votes, axis=1)
        return self.classes[predicted_indices]
    
    def save(self, filepath):
        """保存模型到npz文件"""
        save_dict = {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'classes': self.classes,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
        }
        
        # 保存每个分类器的支持向量
        for key, clf in self.classifiers.items():
            prefix = f"clf_{key[0]}_{key[1]}"
            save_dict[f"{prefix}_support_vectors"] = clf.support_vectors
            save_dict[f"{prefix}_support_labels"] = clf.support_labels
            save_dict[f"{prefix}_support_alpha"] = clf.support_alpha
            save_dict[f"{prefix}_b"] = clf.b
        
        np.savez_compressed(filepath, **save_dict)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """从npz文件加载模型"""
        data = np.load(filepath, allow_pickle=True)
        
        self.C = float(data['C'])
        self.kernel = str(data['kernel'])
        self.gamma = float(data['gamma'])
        self.degree = int(data['degree'])
        self.coef0 = float(data['coef0'])
        self.max_iter = int(data['max_iter'])
        self.tol = float(data['tol'])
        self.classes = data['classes']
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        
        # 加载每个分类器
        self.classifiers = {}
        n_classes = len(self.classes)
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                key = (self.classes[i], self.classes[j])
                prefix = f"clf_{key[0]}_{key[1]}"
                
                clf = BinarySVM(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    max_iter=self.max_iter,
                    tol=self.tol
                )
                
                clf.support_vectors = data[f"{prefix}_support_vectors"]
                clf.support_labels = data[f"{prefix}_support_labels"]
                clf.support_alpha = data[f"{prefix}_support_alpha"]
                clf.b = float(data[f"{prefix}_b"])
                
                self.classifiers[key] = clf
        
        print(f"Model loaded from {filepath}")
