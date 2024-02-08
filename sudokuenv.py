import gymnasium as gym
from gymnasium import spaces
import numpy as np

# This entire board is of size BOARD_COLS * BOARD_ROWS, which is divided into multiples of boxes of size BOX_COLS * BOX_ROWS
BOARD_COLS = 4 # 9
BOARD_ROWS = 4 # 9
BOX_COLS = 2 # 3
BOX_ROWS = 2 # 3

def check_row(board, row_num):
	num_counts = [0] * (BOARD_COLS + 1)
	for i in range(BOARD_COLS):
		num_counts[board[row_num][i]] += 1
		if (board[row_num][i] != 0 and num_counts[board[row_num][i]] > 1): #occured 2+ times in row
			return False
	return True

def check_col(board, col_num):
	num_counts = [0] * (BOARD_ROWS + 1)
	for i in range(BOARD_ROWS):
		num_counts[board[i][col_num]] += 1
		if (board[i][col_num] != 0 and num_counts[board[i][col_num]] > 1): #occured 2+ times in col
			return False
	return True

def check_box(board, row_num, col_num):
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

def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0

	
class SudokuEnv(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self):
		super(SudokuEnv, self).__init__()
		#self.action_space = spaces.Tuple([spaces.Discrete(BOARD_ROWS), spaces.Discrete(BOARD_COLS), spaces.Discrete(BOX_ROWS * BOX_COLS, start=1)])
		self.action_space = spaces.MultiDiscrete([BOARD_ROWS, BOARD_COLS, BOX_ROWS * BOX_COLS])
		# Example: (2, 1, 4) # Selecting row 2, column 2, and placing number 4 there
		self.observation_space = spaces.Box(low=0, high=BOX_ROWS * BOX_COLS, shape=(BOARD_ROWS,BOARD_COLS), dtype=np.int64)
		# 0 is open space. This is a BOARD_ROWS * BOARD_COLS grid of numbers from 0 to BOX_ROWS*BOX_COLS

		self.row_stats = []
		self.col_stats = []
		self.box_stats = []
		self.square_stats = []
		self.reward_stats = []
		self.game_step_counts = []
		self.game_count = 0
		self.success_count = 0
		self.failure_count = 0
		self.current_game_step_count = 0
		self.print_count = 0

	def print_board(self, title="Board:"):
		# debug - print board
		self.print_count += 1
		if title[0]!="S":
			if not is_power_of_2(self.print_count):
				#print(f"Not printing {self.print_count}")
				return


		print(f"{self.print_count}: {title}")
		for i in range(BOARD_ROWS):
			print(self.board[i])
		print()

	def step(self, action):
		# print(action[0], action[1], action[2]) # debug

		# perform action
		action[2] += 1   # because in the action table, the number starts from 0, which maps to 1.
		self.current_game_step_count += 1
		if self.board[action[0]][action[1]] != 0: 
			self.reward -= 50 # punishment for overwriting existing number
			self.failure_count += 1
			self.print_board(f"Failure by overwring existing number through action {action} after {self.current_game_step_count} steps:")
			self.done = True
		else:
			self.board[action[0]][action[1]] = action[2]
		
	        # change reward
			valid_move = check_row(self.board, action[0]) and check_col(self.board, action[1]) and check_box(self.board, action[0], action[1])
			if valid_move == False:
				self.reward -= 40 # punishment for making an invalid move
				self.failure_count += 1
				self.print_board(f"Failure from an invalid move {action} after {self.current_game_step_count} steps:")
				self.done = True
			else:
				if check_row(self.board, action[0]) and row_filled(self.board, action[0]) == True:
					self.reward += 10 # reward for filling in row correctly
				elif check_col(self.board, action[1]) and col_filled(self.board, action[1]) == True: 
					self.reward += 10 # reward for filling in col correctly
				elif check_box(self.board, action[0], action[1]) and box_filled(self.board, action[0], action[1]) == True:
					self.reward += 10 # reward for filling in box correctly
			
        		# check for end game
				# done is true if board satisfies all constraints
				self.done = True
				for i in range(BOARD_ROWS): # all rows OK
					self.done = self.done and row_filled(self.board, i) and check_row(self.board, i)
				for i in range(BOARD_COLS): # all cols OK
					self.done = self.done and col_filled(self.board, i) and check_col(self.board, i) 
				for i in range(BOX_ROWS): # all boxes OK
					for j in range(BOX_COLS):
						self.done = self.done and box_filled(self.board, i, j) and check_box(self.board, i * BOX_ROWS, j * BOX_COLS) 
		
		        # reward for completing board
				if self.done == True:
					self.reward += 200 # reward for completing board
					self.success_count += 1
					self.print_board("Success:")

				# end game after 3 full bad boards
				if board_filled(self.board) == True:
					self.fullbad += 1
				if self.fullbad > 3:
					self.done = True
					self.reward -= 200
					self.failure_count += 1
					# self.print_board("Failure:")

		# good_rows, good_cols, good_boxes, filled_squares = self.get_statistics()
		# print(good_rows, good_cols, good_boxes, filled_squares)
			
		# record stats
		if self.done == True:
			self.post_game_stats()

		# create observation
		action[2] -= 1
		info = {}
		observation = np.array(self.board)
		# obs, reward, terminated, truncated, info
		# terminated = True if env terminates (task completes, failure, etc)
		# truncated = True if episode truncates (time limit, etc)
		return observation, self.reward, self.done, False, info
		
	def reset(self, seed=None): # initializes entire board
		# set up empty board
		self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64) # empty board
		
        # set up reward and done
		self.reward = 0
		self.fullbad = 0
		self.done = False
		self.current_game_step_count = 0
		
        #create observation
		info = {}
		observation = np.array(self.board)
		return observation, info # reward and done can't be included
	
	def post_game_stats(self): # post-game stats collection - this is invoked after every single game becomes done.
		good_rows, good_cols, good_boxes = 0, 0, 0
		filled_squares = 0.0
		for i in range(BOARD_ROWS):
			if row_filled(self.board, i) and check_row(self.board, i) == True:
				good_rows += 1
		for i in range(BOARD_COLS):
			if col_filled(self.board, i) and check_col(self.board, i) == True:
				good_cols += 1
		for i in range(BOX_ROWS):
			for j in range(BOX_COLS):
				if box_filled(self.board, i * BOX_ROWS, j * BOX_COLS) and check_box(self.board, i * BOX_ROWS, j * BOX_COLS):
					good_boxes += 1
		for i in range(BOARD_ROWS):
			for j in range(BOARD_COLS):
				if self.board[i][j] != 0:
					filled_squares += 1.0
		filled_squares /= (BOARD_COLS * BOARD_ROWS)

		self.row_stats.append(good_rows)
		self.col_stats.append(good_cols)
		self.box_stats.append(good_boxes)
		self.square_stats.append(filled_squares)
		self.reward_stats.append(self.reward)
		self.game_step_counts.append(self.current_game_step_count)
		self.game_count += 1
		self.current_game_step_count = 0

	def get_statistics(self):
		tmp_row_stats, tmp_col_stats, tmp_box_stats, tmp_square_stats, tmp_reward_stats, tmp_game_step_count = 0, 0, 0, 0, 0, 0
		for i in range(self.game_count):
			tmp_row_stats += self.row_stats[i]
			tmp_col_stats += self.col_stats[i]
			tmp_box_stats += self.box_stats[i]
			tmp_square_stats += self.square_stats[i]
			tmp_reward_stats += self.reward_stats[i]
			tmp_game_step_count += self.game_step_counts[i]

		tmp_row_stats /= self.game_count
		tmp_col_stats /= self.game_count
		tmp_box_stats /= self.game_count
		tmp_square_stats /= self.game_count
		tmp_reward_stats /= self.game_count
		tmp_game_step_count /= self.game_count
		tmp_game_count = self.game_count

		self.row_stats = []
		self.col_stats = []
		self.box_stats = []
		self.square_stats = []
		self.reward_stats = []
		self.game_count = 0

		return tmp_row_stats, tmp_col_stats, tmp_box_stats, tmp_square_stats, tmp_reward_stats, tmp_game_count, self.success_count, self.failure_count, tmp_game_step_count
	
	# def close(self):
	# 	...