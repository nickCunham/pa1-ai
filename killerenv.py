import gymnasium
from gymnasium import spaces

BOARD_COLS = 9
BOARD_ROWS = 9
BOX_COLS = 3
BOX_ROWS = 3

def check_row(board, row_num):
	num_counts = [] * BOARD_COLS
	for i in range(BOARD_COLS):
		num_counts[board[row_num][i]] += 1
		if (num_counts[board[row_num][i]] > 1): #occured 2+ times in row
			return False
	return True

def check_col(board, col_num):
	num_counts = [] * BOARD_ROWS
	for i in range(BOARD_ROWS):
		num_counts[board[i][col_num]] += 1
		if (num_counts[board[i][col_num]] > 1): #occured 2+ times in col
			return False
	return True

def check_box(board, row_num, col_num):
	box_start_row = row_num // BOX_ROWS
	box_start_col = col_num // BOX_COLS
	num_counts = [] * BOARD_COLS
	for i in range(BOX_ROWS):
		for j in range(BOX_COLS):
			num_counts[board[box_start_row + i][box_start_col + j]] += 1
			if (num_counts[board[box_start_row + i][box_start_col + j]] > 1): #occured 2+ times in box
				return False
	return True

	
class KillerEnv(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self):
		super(KillerEnv, self).__init__()
		self.action_space = spaces.Tuple(spaces.Discrete(BOARD_ROWS), spaces.Discrete(BOARD_COLS), spaces.Discrete(BOX_ROWS * BOX_COLS, start=1))
		self.observation_space = spaces.Box(low=0, high=BOX_ROWS * BOX_COLS, shape=(BOARD_ROWS,BOARD_COLS), dtype=np.uint8)

	# def step(self, action):
	# 	...
	# 	return observation, reward, done, info
	# def reset(self):
	# 	...
	# 	return observation  # reward, done, info can't be included
	# def render(self, mode='human'):
	# 	...
	# def close (self):
	# 	...