import torch
import numpy as np
import os
from model import RewardPredictor

def export():
    device = torch.device("cpu")
    model = RewardPredictor()
    model_path = 'model.pth'
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    weights = {}
    
    # LSTM weights
    # Layer 0
    weights['lstm_l0_ih_w'] = model.lstm.weight_ih_l0.detach().numpy()
    weights['lstm_l0_hh_w'] = model.lstm.weight_hh_l0.detach().numpy()
    weights['lstm_l0_ih_b'] = model.lstm.bias_ih_l0.detach().numpy()
    weights['lstm_l0_hh_b'] = model.lstm.bias_hh_l0.detach().numpy()
    
    # Layer 1
    weights['lstm_l1_ih_w'] = model.lstm.weight_ih_l1.detach().numpy()
    weights['lstm_l1_hh_w'] = model.lstm.weight_hh_l1.detach().numpy()
    weights['lstm_l1_ih_b'] = model.lstm.bias_ih_l1.detach().numpy()
    weights['lstm_l1_hh_b'] = model.lstm.bias_hh_l1.detach().numpy()
    
    # Head weights
    # Linear 1
    weights['head_0_w'] = model.head[0].weight.detach().numpy()
    weights['head_0_b'] = model.head[0].bias.detach().numpy()
    
    # Linear 2
    weights['head_2_w'] = model.head[2].weight.detach().numpy()
    weights['head_2_b'] = model.head[2].bias.detach().numpy()
    
    # Linear 3
    weights['head_4_w'] = model.head[4].weight.detach().numpy()
    weights['head_4_b'] = model.head[4].bias.detach().numpy()
    
    np.savez('model_weights.npz', **weights)
    print("Weights exported to model_weights.npz")

if __name__ == "__main__":
    export()
