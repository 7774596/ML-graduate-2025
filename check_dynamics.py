import pandas as pd
import numpy as np

def get_next_position(p, action):
    p_new = p + action
    if p_new < 0:
        return 0 + abs(p_new) // 2
    elif p_new > 99:
        return 99 - (p_new - 99) // 2
    else:
        return p_new

def check_dynamics():
    df = pd.read_csv('offline_data.csv')
    
    matches = 0
    total = 0
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        if row['trial_id'] != next_row['trial_id']:
            continue
            
        p = row['P']
        action = row['action']
        expected_next_p = get_next_position(p, action)
        
        actual_next_p = next_row['P']
        
        if expected_next_p == actual_next_p:
            matches += 1
        total += 1
        
        if total < 10:
            print(f"P: {p}, A: {action} -> Exp: {expected_next_p}, Act: {actual_next_p}")

    print(f"Matches: {matches}/{total} ({matches/total*100:.2f}%)")

if __name__ == "__main__":
    check_dynamics()
