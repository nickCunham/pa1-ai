from stable_baselines3 import PPO
import os
import time
import wandb

from sudokuenv import SudokuEnv

wandb.init(
    project="sudoku-project",
    name = "Sudoku " + wandb.util.generate_id(),
    tags=["sudoku", ],	
    config={
	    "monitor_gym": True
    }
)
models_dir = f"models/{int(time.time())}/" # f"models/1704225155/"
logdir = f"logs/{int(time.time())}/" # f"logs/1704225155/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SudokuEnv()
env.reset()

# model_path = f"{models_dir}/400000"

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir) # PPO.load(model_path, env=env) # 

TIMESTEPS = 10000
iters = 0 
while iters < 100000:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	good_rows, good_cols, good_boxes, filled_squares, reward, game_count, success_count, failure_count, game_step_count = env.get_statistics()
	wandb.log({"good_rows": good_rows, "good_cols": good_cols, "good_boxes": good_boxes, "filled_squares": filled_squares, 
			"reward": reward, "game_count": game_count, "success": success_count, "failure": failure_count, "ave steps per game": game_step_count } )
	