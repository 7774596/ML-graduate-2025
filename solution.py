import pandas as pd
import numpy as np
from model import Model


class Solution:
    def __init__(self):
        self.model = Model()
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        self.string_columns = ['protocol_type', 'service', 'flag']
        self.encoders = {}

    def _encode_features(self, X_df, fit_mode=False):
        X_df = X_df.copy()

        for col in self.string_columns:
            if col in X_df.columns:
                if fit_mode:
                    # Handle potential non-string values in string columns
                    uniques = X_df[col].astype(str).unique()
                    self.encoders[col] = {val: idx for idx, val in enumerate(uniques)}
                
                # Map and fillna
                # Ensure we map strings
                X_df[col] = X_df[col].astype(str).map(self.encoders.get(col, {})).fillna(0)

        X = X_df.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        return X

    def fit(self, X_df, y, learning_rate=1.0, epochs=1):
        X = self._encode_features(X_df, fit_mode=True)
        self.model.fit(X, y, learning_rate, epochs)

    def forward(self, sample: dict) -> dict:
        # Optimized forward for single sample using on-demand feature extraction
        # We pass the raw sample dict to the model, which only extracts features needed by the tree
        probability = self.model.predict_single(sample, self.feature_names, self.encoders)

        return {
            'prediction': int(probability >= 0.5),
            'probability': float(probability)
        }