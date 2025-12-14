from solution import Solution
import numpy as np

def test():
    sol = Solution()
    state = 50
    trajectory = []
    
    action = sol.policy(state, trajectory)
    print(f"Step 0: State {state}, Action {action}")
    
    # Simulate a step
    next_state = sol.get_next_position(state, action)
    reward = 0.5 # Fake reward
    trajectory.append((next_state, action, reward))
    
    action = sol.policy(next_state, trajectory)
    print(f"Step 1: State {next_state}, Action {action}")

if __name__ == "__main__":
    test()
