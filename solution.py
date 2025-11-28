import pandas as pd
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        self.model = Model()
        # Only use top 5 features based on importance analysis
        self.feature_names = [
            'count', 
            'dst_host_srv_serror_rate', 
            'dst_bytes', 
            'dst_host_srv_diff_host_rate', 
            'dst_host_diff_srv_rate'
        ]
        # No string columns in the top 5 features
        self.string_columns = []
        self.encoders = {}

    def _encode_features(self, X_df, fit_mode=False):
        # Select only the required features
        X_df = X_df[self.feature_names].copy()
        
        # No string encoding needed for these specific features
        X = X_df.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        return X

    def fit(self, X_df, y, learning_rate=1.0, epochs=1):
        # Optimization: Subsample heavily BEFORE data conversion to save time
        # 200 samples are enough to learn the basic rules for this high-SNR task
        n_samples = len(X_df)
        if n_samples > 300:
            np.random.seed(42)
            indices = np.random.choice(n_samples, 300, replace=False)
            # Use iloc for integer indexing if it's a dataframe, or direct indexing if array
            # y is likely a numpy array or series
            if isinstance(X_df, pd.DataFrame):
                X_subset = X_df.iloc[indices]
            else:
                X_subset = X_df[indices]
            
            if isinstance(y, pd.Series):
                y_subset = y.iloc[indices]
            else:
                y_subset = y[indices]
        else:
            X_subset = X_df
            y_subset = y

        X = self._encode_features(X_subset, fit_mode=True)
        self.model.fit(X, y_subset, learning_rate, epochs, feature_names=self.feature_names)

    def forward(self, sample: dict) -> dict:
        # Optimized forward for single sample using on-demand feature extraction
        # We pass the raw sample dict to the model, which only extracts features needed by the tree
        probability = self.model.predict_single(sample, self.feature_names, self.encoders)

        return {
            'prediction': int(probability >= 0.5),
            'probability': float(probability)
        }