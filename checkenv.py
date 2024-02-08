#from tutorial at https://pythonprogramming.net/custom-environment-reinforcement-learning-stable-baselines-3-tutorial/

from stable_baselines3.common.env_checker import check_env
# from killerenv import KillerEnv
from sudokuenv import SudokuEnv

#env = KillerEnv()
env = SudokuEnv()
check_env(env) # checks custom environment, outputs additional warnings if needed