# This is a self-contained program to demonstrate a simple grid-based RL solution for a small Sudoku.
# Training result is at: https://wandb.ai/liu-chang/Grid%204

import gymnasium as gym
from   gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

import os
import time
import wandb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

hostname = os.uname()[1]

wandb.init(
    project="Grid 4",
    name = hostname + ": " + wandb.util.generate_id(),
    tags=["grid4", ],	
    config={
	    "monitor_gym": True
    }
)

models_dir = f"models/{int(time.time())}/" 
logdir = f"logs/{int(time.time())}/" 

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


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

def check_col_complete(board, col_num):
	num_counts = [0] * (BOARD_ROWS + 1)
	for i in range(BOARD_ROWS):
		num_counts[board[i][col_num]] += 1
		if (board[i][col_num] == 0 or num_counts[board[i][col_num]] > 1): #occured 2+ times in col
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

def row_filled(board, row_num):
	for i in range(BOARD_COLS):
		if board[row_num][i] == 0:
			return False
	return True

def col_filled(board, col_num):
	for i in range(BOARD_ROWS):
		if board[i][col_num] == 0: 
			return False
	return True

def box_filled(board, row_num, col_num):
	box_start_row = row_num // BOX_ROWS
	box_start_col = col_num // BOX_COLS
	for i in range(BOX_ROWS):
		for j in range(BOX_COLS):
			if board[box_start_row + i][box_start_col + j] == 0:
				return False
	return True

def board_filled(board):
	for i in range(BOARD_ROWS):
		for j in range(BOARD_COLS):
			if board[i][j] == 0:
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
        reward = 1
        done = False
        info = {}
        new_state = self.state  
        if new_state[action[0]][action[1]] != 0 : # assignment on already assigned spot
            reward = -20
            done = True
            return self.state, reward, done, False, {}
        
        new_state[action[0]][action[1]] = action[2] + 1  # Modify the data       (0 is fill in 1)

        valid_move = check_row_conflict(self.state, action[0]) and check_col_conflict(self.state, action[1]) and check_box_conflict(self.state, action[0], action[1])
        if not valid_move :
            reward = -10
            done = True
            return self.state, reward, done, False, {}

        # Check for completed rows or cols
        for i in range(BOARD_ROWS):
            if check_row_complete(new_state,i):
                reward = 2
                print(f"ROW {i} is complete. {new_state[i]}\n{new_state}")
                break

        for i in range(BOARD_COLS):
            if check_col_complete(new_state,i):
                reward = 2
                print(f"COL {i} is complete. {new_state[:,i]}\n{new_state}")
                break

		# check for end game
		# done is true if board satisfies all constraints
        done = True
        for i in range(BOARD_ROWS): # all rows OK
            done = done and row_filled(new_state, i) and check_row_conflict(new_state, i)
            if not done:
                break
        if done:
            reward += 10
            print(f"All Rows are complete. \n{new_state}")
            for i in range(BOARD_COLS): # all cols OK
                done = done and col_filled(new_state, i) and check_col_conflict(new_state, i) 
                if not done:
                      break
        if done:
            reward += 10
            print(f"All Columns are complete. \n{new_state}")
            for i in range(BOX_ROWS): # all boxes OK
                for j in range(BOX_COLS):
                    done = done and box_filled(new_state, i, j) and check_box_conflict(new_state, i * BOX_ROWS, j * BOX_COLS) 
                    if not done:
                        break
        if done: 
            reward += 40
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
eval_callback = EvalCallback(eval_env=env, n_eval_episodes=10, verbose=1)

print(f"Attributes available from the model are: {dir(model)}")
# The print out will be something list this: ['__abstractmethods__', '__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_current_progress_remaining', '_custom_logger', '_episode_num', '_excluded_save_params', '_get_policy_from_name', '_get_torch_save_params', '_init_callback', '_last_episode_starts', '_last_obs', '_last_original_obs', '_logger', '_n_updates', '_num_timesteps_at_start', '_setup_learn', '_setup_lr_schedule', '_setup_model', '_stats_window_size', '_total_timesteps', '_update_current_progress_remaining', '_update_info_buffer', '_update_learning_rate', '_vec_normalize_env', '_wrap_env', 'action_noise', 'action_space', 'batch_size', 'clip_range', 'clip_range_vf', 'collect_rollouts', 'device', 'ent_coef', 'env', 'ep_info_buffer', 'ep_success_buffer', 'gae_lambda', 'gamma', 'get_env', 'get_parameters', 'get_vec_normalize_env', 'learn', 'learning_rate', 'load', 'logger', 'lr_schedule', 'max_grad_norm', 'n_envs', 'n_epochs', 'n_steps', 'normalize_advantage', 'num_timesteps', 'observation_space', 'policy', 'policy_aliases', 'policy_class', 'policy_kwargs', 'predict', 'rollout_buffer', 'rollout_buffer_class', 'rollout_buffer_kwargs', 'save', 'sde_sample_freq', 'seed', 'set_env', 'set_logger', 'set_parameters', 'set_random_seed', 'start_time', 'target_kl', 'tensorboard_log', 'train', 'use_sde', 'verbose', 'vf_coef']

# Train the model
TIMESTEPS = 10_000
iters = 0 
while iters < 100000:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=[eval_callback])
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	print("Evaluation reward:", eval_callback.best_mean_reward)
	wandb.log({"best_mean_reward": eval_callback.best_mean_reward }) 

print(f"Now testing the model ...")

# Test the model
obs, _info = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render(action=action, step=i, reward=reward)  # Optional: Render the environment
    if done and reward > 0:
        print(f"Goal reached after {i+1} steps! Reward: {reward}")
        break
    if done:
        env.reset()