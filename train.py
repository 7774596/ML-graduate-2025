import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import RewardPredictor
import os

class OfflineDataset(Dataset):
    def __init__(self, csv_file, context_len=30):
        self.df = pd.read_csv(csv_file)
        self.context_len = context_len
        self.sequences = []
        self.targets = []
        
        # Group by trial
        grouped = self.df.groupby('trial_id')
        
        print("Processing data...")
        # Use a list comprehension for speed
        all_sequences = []
        all_targets = []
        
        for trial_id, group in grouped:
            # Sort by step_id
            group = group.sort_values('step_id')
            p_values = group['P'].values / 99.0 # Normalize P
            r_values = group['reward'].values
            
            # Vectorized sliding window creation would be faster but let's stick to loop for clarity
            # We can optimize if needed.
            
            L = len(p_values)
            for i in range(L - 1):
                # History: up to i
                # Target: i+1
                
                # We want fixed size context
                start = max(0, i + 1 - self.context_len)
                end = i + 1
                
                hist_p = p_values[start:end]
                hist_r = r_values[start:end]
                
                next_p = p_values[i+1]
                target_r = r_values[i+1]
                
                # Pad
                if len(hist_p) < self.context_len:
                    pad = self.context_len - len(hist_p)
                    hist_p = np.pad(hist_p, (pad, 0), 'constant')
                    hist_r = np.pad(hist_r, (pad, 0), 'constant')
                
                all_sequences.append((hist_p, hist_r, next_p))
                all_targets.append(target_r)
        
        self.sequences = all_sequences
        self.targets = all_targets
        print(f"Data processed. {len(self.sequences)} samples.")
                
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        hist_p, hist_r, next_p = self.sequences[idx]
        target_r = self.targets[idx]
        
        return (
            torch.FloatTensor(hist_p).unsqueeze(-1),
            torch.FloatTensor(hist_r).unsqueeze(-1),
            torch.FloatTensor([next_p]),
            torch.FloatTensor([target_r])
        )

def train():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = OfflineDataset('offline_data.csv', context_len=30)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True) # Larger batch size
    
    model = RewardPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            hist_p, hist_r, next_p, target_r = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            pred_r = model(hist_p, hist_r, next_p)
            loss = criterion(pred_r, target_r)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()
