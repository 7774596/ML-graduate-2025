import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv('offline_data.csv')
    print("Reward stats:")
    print(df['reward'].describe())
    
    # Check if reward is symmetric or has a pattern
    # Let's try to guess the reward function.
    # Hypothesis 1: R = A - B * |P - q|
    # Hypothesis 2: R = A * exp(-B * |P - q|^2)
    # Hypothesis 3: R = cos(something)
    
    # Let's look at the max possible reward.
    max_reward = df['reward'].max()
    print(f"Max reward: {max_reward}")
    
    # If max reward corresponds to distance 0.
    # Let's assume at some point P approx q.
    
    # Let's try to see if we can find a q for each step.
    # If R = f(|P - q|), then |P - q| = f_inv(R).
    # So |P - q| is constant for a given R.
    # This means P - q = +/- D. => q = P +/- D.
    
    # Let's try to infer D from R.
    # If we plot R vs P, it won't make sense because q changes.
    # But if we assume q moves slowly or follows a pattern...
    
    # Let's look at the difference in P and difference in R? No.
    
    # Let's try to fit a simple model for Reward.
    # Maybe R is just a function of P and q?
    # What if q is fixed in some trials? (Unlikely "dynamic position optimization")
    
    # Let's try to see if we can predict Reward from P and some hidden state.
    
    # Filter for high rewards
    high_reward_df = df[df['reward'] > 1.5]
    print(f"Number of high reward steps: {len(high_reward_df)}")
    
    if len(high_reward_df) > 0:
        print("Sample high reward steps:")
        print(high_reward_df.head(10))
        
        # Let's look at one trial with high rewards
        trial_id = high_reward_df.iloc[0]['trial_id']
        trial_data = df[df['trial_id'] == trial_id]
        
        print(f"\nAnalysis of Trial {trial_id}:")
        # Print steps where reward is high
        print(trial_data[trial_data['reward'] > 1.0][['step_id', 'P', 'action', 'reward']])
        
        # Check if P changes in a pattern when reward is high
        # If P is constant and reward is high, then q is constant.
        # If P changes and reward stays high, q is moving with P.
        pass

if __name__ == "__main__":
    analyze()
