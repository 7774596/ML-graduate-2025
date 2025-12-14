import torch
import torch.nn as nn

class RewardPredictor(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # Input: P (1), R (1). Total 2.
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True, num_layers=2)
        # Head: Hidden (H) + Next P (1) -> Reward (1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, history_p, history_r, next_p):
        # history_p: (B, L, 1)
        # history_r: (B, L, 1)
        # next_p: (B, 1) or (B, K, 1)
        
        x = torch.cat([history_p, history_r], dim=-1) # (B, L, 2)
        out, (h, c) = self.lstm(x)
        
        # Take the last hidden state
        last_h = out[:, -1, :] # (B, H)
        
        if next_p.dim() == 3:
            # next_p is (B, K, 1)
            k = next_p.size(1)
            last_h_exp = last_h.unsqueeze(1).expand(-1, k, -1)
            inp = torch.cat([last_h_exp, next_p], dim=-1)
        else:
            # next_p is (B, 1)
            inp = torch.cat([last_h, next_p], dim=-1)
            
        pred_r = self.head(inp)
        return pred_r
