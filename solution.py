import os
import numpy as np
from model import Model


class Solution:
    """Inference-only wrapper for evaluation.

    仅负责加载已训练好的 SVM 模型 (npz) 并提供 forward 接口。
    训练请使用 `train.py`：
        python train.py --kernel rbf --C 2.0 --output svm_model.npz

    环境变量：
        MODEL_PATH: 指定模型文件路径 (默认 'svm_model.npz')
    """

    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "svm_model.npz")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please run train.py first to generate it."
            )
        self.model = Model.load(model_path)

    def forward(self, sample):
        x = sample["image"] if isinstance(sample, dict) else sample
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(x)
        return {"prediction": int(pred[0])}
