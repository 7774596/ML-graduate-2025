import numpy as np
import os
from model import Model


class Solution:
    """
    SVM推理接口类
    
    用于加载训练好的SVM模型并进行预测
    """
    
    def __init__(self, model_path='svm_improved_model.npz'):
        """
        初始化推理类,加载预训练模型
        
        参数:
            model_path: 模型文件路径 (.npz格式)
        """
        self.model = Model(verbose=False)
        self.use_feature_engineering = False  # 是否使用特征工程
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 如果指定的模型不存在,尝试训练一个默认模型
            print(f"Model file '{model_path}' not found.")
            print("Training a default SVM model...")
            
            try:
                data = np.load("data/train.npz")
                X_train = data["X_train"]
                y_train = data["y_train"]
                
                # 训练一个快速的RBF-SVM模型
                self.model = Model(
                    C=2.0,
                    kernel='rbf',
                    gamma=0.005,
                    max_iter=100,
                    tol=1e-3,
                    verbose=True
                )
                self.model.fit(X_train, y_train)
                
                # 保存模型
                self.model.save(model_path)
                print(f"Model trained and saved to '{model_path}'")
                
            except Exception as e:
                raise RuntimeError(f"Failed to train default model: {e}")
        else:
            # 加载已存在的模型
            print(f"Loading model from '{model_path}'...")
            self.model.load(model_path)
            print("Model loaded successfully!")
            
            # 检测是否需要特征工程（根据模型的特征维度）
            # 标准特征维度是28*28=784，如果是2352则使用了特征工程
            if hasattr(self.model, 'X_mean') and self.model.X_mean is not None:
                if len(self.model.X_mean) > 784:
                    self.use_feature_engineering = True
                    print(f"Detected feature engineering (dim={len(self.model.X_mean)})")
    
    def _apply_feature_engineering(self, X):
        """
        应用特征工程（与train_optimized.py中的feature_engineering一致）
        """
        X_flat = X.reshape(X.shape[0], -1)
        
        # 基础特征: 标准化后的像素值
        features = [X_flat]
        
        # 特征1: 像素值的平方(增强对比度信息)
        features.append(X_flat ** 2)
        
        # 特征2: 平方根(压缩动态范围)
        features.append(np.sqrt(X_flat))
        
        # 组合所有特征
        X_enhanced = np.hstack(features)
        
        return X_enhanced
    
    def forward(self, sample):
        """
        模型推理接口,接收单条样本数据并返回预测结果
        
        参数:
            sample: numpy数组或字典
                   - 如果是字典: {'image': numpy数组, shape (H, W)}
                   - 如果是数组: shape (H, W)
        
        返回:
            dict: {'prediction': int}, 预测类别 (0-9)
        """
        # 提取图像数据
        if isinstance(sample, dict):
            x = sample["image"]
        else:
            x = sample
        
        # 确保数据格式正确
        x = np.asarray(x, dtype=np.float32)
        
        # reshape为(1, H*W)格式
        if x.ndim == 2:
            x = x.reshape(1, -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        
        # 如果模型使用了特征工程，应用相同的特征工程
        if self.use_feature_engineering:
            x = self._apply_feature_engineering(x)
        
        # 预测
        prediction = self.model.predict(x)
        
        return {"prediction": int(prediction[0])}
