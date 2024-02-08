# This is a self-contained program to demonstrate a simple grid-based RL loop with a simple yet meaningful goal.
# Goal: Have one complete row!

import gymnasium as gym
from   gymnasium import spaces
from stable_baselines3 import PPO
import numpy as np

# The entire board is of size BOARD_COLS * BOARD_ROWS, which is divided into multiples of boxes of size BOX_COLS * BOX_ROWS
BOARD_COLS = 4 # 9
BOARD_ROWS = 4 # 9
BOX_COLS = 2 # 3
BOX_ROWS = 2 # 3

def check_row_conflict(board, row_num): # Return True only if there's no conflict
	num_counts = [0] * (BOARD_COLS + 1)
	for i in range(BOARD_COLS):
		num_counts[board[row_num][i]] += 1
		if (board[row_num][i] != 0 and num_counts[board[row_num][i]] > 1): #occured 2+ times in row
			return False
	return True

def check_row_complete(board, row_num): # Return True only if the row is complete
	num_counts = [0] * (BOARD_COLS + 1)
	for i in range(BOARD_COLS):
		num_counts[board[row_num][i]] += 1
		if (board[row_num][i] == 0 or num_counts[board[row_num][i]] > 1): #incomplete or occured 2+ times in row
			return False
	return True

def check_col_conflict(board, col_num):
	num_counts = [0] * (BOARD_ROWS + 1)
	for i in range(BOARD_ROWS):
		num_counts[board[i][col_num]] += 1
		if (board[i][col_num] != 0 and num_counts[board[i][col_num]] > 1): #occured 2+ times in col
			return False
	return True

def check_box_conflict(board, row_num, col_num):
	box_start_row = row_num // BOX_ROWS
	box_start_col = col_num // BOX_COLS
	num_counts = [0] * (BOARD_COLS + 1)
	for i in range(BOX_ROWS):
		for j in range(BOX_COLS):
			num_counts[board[box_start_row + i][box_start_col + j]] += 1
			if (board[box_start_row + i][box_start_col + j] != 0 and num_counts[board[box_start_row + i][box_start_col + j]] > 1): #occured 2+ times in box
				return False
	return True

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
        reward = 0
        done = False
        info = {}
        new_state = self.state  
        if new_state[action[0]][action[1]] != 0 : # duplicate assignment
            done = True
            reward = -2
            return self.state, reward, done, False, {}
        
        new_state[action[0]][action[1]] = action[2] + 1  # Modify the data       (0 is fill in 1)

        valid_move = check_row_conflict(self.state, action[0]) and check_col_conflict(self.state, action[1]) and check_box_conflict(self.state, action[0], action[1])
        if not valid_move :
            done = True
            reward = -1
            return self.state, reward, done, False, {}


        # Check for goal state
        for i in range(BOARD_ROWS):
            if check_row_complete(new_state,i):
                reward = 1
                print(f"Row {i} is complete. {new_state[i]}")
                done = True
                break

        self.state = new_state
        return self.state, reward, done, False, info
 
    def render(self, action=(0,0,0), step=0, reward=0):
        action[2] += 1
        print(f"Rendering: {step}: {self.observation_space} - action: {self.action_space}:{action} - reward: {reward}")
        printout = f""
        for i in range(BOARD_ROWS):
            printout += f"{self.state[i]} <br>\n"
        print(printout)

# Create the environment
env = GridEnv()

# Create the RL model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100_000, log_interval=10_000)

print(f"Now testing the model ...")

# Test the model
obs, _info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render(action=action, step=i, reward=reward)  # Optional: Render the environment
    if done and reward > 0:
        print(f"Goal reached after {i+1} steps! Reward: {reward}")
        break
    if done:
        env.reset()