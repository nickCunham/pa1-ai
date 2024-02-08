# This is a self-contained program to demonstrate a simple grid-based RL loop
# Goal: cell [1][1] must be 3.

import gymnasium as gym
from   gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np

# The entire board is of size BOARD_COLS * BOARD_ROWS, which is divided into multiples of boxes of size BOX_COLS * BOX_ROWS
BOARD_COLS = 4 # 9
BOARD_ROWS = 4 # 9
BOX_COLS = 2 # 3
BOX_ROWS = 2 # 3

# Create a custom grid-based environment
class GridEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=BOX_ROWS * BOX_COLS, shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int64)
        self.action_space = spaces.MultiDiscrete([BOARD_ROWS, BOARD_COLS, BOX_ROWS * BOX_COLS])
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64) 
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64) # empty board
        return self.state, {} # Return observation and info
    
    def step(self, action):
        # Move based on the action
        new_state = self.state  
        new_state[action[0]][action[1]] = action[2]  # Modify the data       

        # Check for conditions and assign reward
        if new_state[0][0] > 2 or new_state[0][1] >= 3 :
            reward = -1  # Penalty for something
        else:
            reward = 0

        # Check for goal state
        if new_state[1][1] == 3:
            reward = 1  # Reward for reaching the goal
            done = True
        else:
            done = False

        # Generate action mask based on current state
        action_mask = np.zeros((BOARD_ROWS, BOARD_COLS, BOX_ROWS * BOX_COLS), dtype=np.int64)  # Initialize mask with all zeros
        action_mask[1][1][2] = 1  # Set valid actions to 1 in the mask

        info = {'action_mask': action_mask}
        self.state = new_state
        return self.state, reward, done, False, info
 
    def render(self, action=(0,0,0), step=0):
        print(f"Rendering: {step}: {self.observation_space} - action: {self.action_space}:{action}")
        printout = f""
        for i in range(BOARD_ROWS):
            printout += f"{self.state[i]} <br>\n"
        print(printout)

# Create the environment
env = GridEnv()

# Create the RL model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the model
print(f"Now testing the model ...")
obs, _info = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render(action=action, step=i)  # Optional: Render the environment
    if done:
        print(f"Goal reached after {i+1} steps!")
        break