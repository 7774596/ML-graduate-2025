# solution.py
import numpy as np
import os
import sys

class NumpyRewardPredictor:
    def __init__(self, weights_path):
        self.weights = np.load(weights_path)
        self.hidden_size = 128
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def tanh(self, x):
        return np.tanh(x)
        
    def relu(self, x):
        return np.maximum(0, x)
        
    def lstm_cell(self, x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        # x: (batch, input_size)
        # h_prev: (batch, hidden_size)
        # c_prev: (batch, hidden_size)
        
        # Gates
        gates = np.dot(x, w_ih.T) + b_ih + np.dot(h_prev, w_hh.T) + b_hh
        
        # Split gates
        i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=1)
        
        i = self.sigmoid(i_gate)
        f = self.sigmoid(f_gate)
        g = self.tanh(g_gate)
        o = self.sigmoid(o_gate)
        
        c_next = f * c_prev + i * g
        h_next = o * self.tanh(c_next)
        
        return h_next, c_next
        
    def forward(self, hist_p, hist_r, next_p):
        # hist_p: (1, L, 1)
        # hist_r: (1, L, 1)
        # next_p: (1, K, 1)
        
        # Prepare input for LSTM
        # Concatenate P and R
        x = np.concatenate([hist_p, hist_r], axis=-1) # (1, L, 2)
        batch_size, seq_len, _ = x.shape
        
        # LSTM Layer 0
        h0 = np.zeros((batch_size, self.hidden_size))
        c0 = np.zeros((batch_size, self.hidden_size))
        
        # We need to process the sequence
        # For simplicity, since batch=1, we can just loop
        # But to be correct with dimensions, let's be careful.
        
        # Layer 0
        h_seq_l0 = []
        h = h0
        c = c0
        for t in range(seq_len):
            xt = x[:, t, :]
            h, c = self.lstm_cell(xt, h, c, 
                                  self.weights['lstm_l0_ih_w'], self.weights['lstm_l0_hh_w'], 
                                  self.weights['lstm_l0_ih_b'], self.weights['lstm_l0_hh_b'])
            h_seq_l0.append(h)
            
        h_seq_l0 = np.stack(h_seq_l0, axis=1) # (1, L, H)
        
        # Layer 1
        h1 = np.zeros((batch_size, self.hidden_size))
        c1 = np.zeros((batch_size, self.hidden_size))
        
        h = h1
        c = c1
        for t in range(seq_len):
            xt = h_seq_l0[:, t, :]
            h, c = self.lstm_cell(xt, h, c, 
                                  self.weights['lstm_l1_ih_w'], self.weights['lstm_l1_hh_w'], 
                                  self.weights['lstm_l1_ih_b'], self.weights['lstm_l1_hh_b'])
            
        # Last hidden state from Layer 1
        last_h = h # (1, H)
        
        # Head
        # next_p is (1, K, 1)
        K = next_p.shape[1]
        
        # Expand last_h to match K
        last_h_exp = np.repeat(last_h[:, np.newaxis, :], K, axis=1) # (1, K, H)
        
        # Concatenate
        inp = np.concatenate([last_h_exp, next_p], axis=-1) # (1, K, H+1)
        
        # Flatten for Linear layers (or just use matmul broadcasting)
        inp_flat = inp.reshape(-1, self.hidden_size + 1)
        
        # Linear 1
        out = np.dot(inp_flat, self.weights['head_0_w'].T) + self.weights['head_0_b']
        out = self.relu(out)
        
        # Linear 2
        out = np.dot(out, self.weights['head_2_w'].T) + self.weights['head_2_b']
        out = self.relu(out)
        
        # Linear 3
        out = np.dot(out, self.weights['head_4_w'].T) + self.weights['head_4_b']
        
        return out.reshape(batch_size, K)

class Solution:
    def __init__(self):
        # 设置模式：0=不提交到排行榜，1=提交到排行榜,默认为1
        self.mode = 0
        
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights.npz')
        if os.path.exists(weights_path):
            try:
                self.model = NumpyRewardPredictor(weights_path)
                print("Model loaded successfully (Numpy).")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("Warning: model_weights.npz not found!")
            self.model = None
        
        self.context_len = 30

    def get_next_position(self, p, action):
        p_new = p + action
        if p_new < 0:
            return 0 + abs(p_new) // 2
        elif p_new > 99:
            return 99 - (p_new - 99) // 2
        else:
            return p_new

    def policy(self, state, trajectory):
        """
        实现你的策略函数
        
        参数:
            state: 当前状态（智能体位置P）
            trajectory: 历史轨迹列表，每个元素为(P, action, reward)三元组
        
        返回:
            action: 要执行的动作（整数，范围-99到99）
        """
        
        if self.model is None:
            return np.random.randint(-99, 99)
            
        # Prepare history
        if len(trajectory) == 0:
            hist_p = []
            hist_r = []
        else:
            # trajectory contains (P, action, reward)
            # We use P and reward
            hist_p = [t[0] / 99.0 for t in trajectory]
            hist_r = [t[2] for t in trajectory]
            
        # Pad
        if len(hist_p) < self.context_len:
            pad = self.context_len - len(hist_p)
            hist_p = [0.0] * pad + hist_p
            hist_r = [0.0] * pad + hist_r
        else:
            hist_p = hist_p[-self.context_len:]
            hist_r = hist_r[-self.context_len:]
            
        # Convert to numpy
        hist_p_np = np.array(hist_p, dtype=np.float32).reshape(1, self.context_len, 1)
        hist_r_np = np.array(hist_r, dtype=np.float32).reshape(1, self.context_len, 1)
        
        # Generate candidates
        actions = np.arange(-99, 100)
        candidates_p = []
        
        for a in actions:
            next_p = self.get_next_position(state, a)
            candidates_p.append(next_p / 99.0)
            
        candidates_p_np = np.array(candidates_p, dtype=np.float32).reshape(1, len(actions), 1)
        
        # Predict
        pred_rewards = self.model.forward(hist_p_np, hist_r_np, candidates_p_np) # (1, 199)
        
        pred_rewards = pred_rewards.flatten()
        
        # Pick best
        best_idx = np.argmax(pred_rewards)
        best_action = actions[best_idx]
        
        return int(best_action)
