from sudokuenv import SudokuEnv


env = SudokuEnv()
episodes = 2

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, truncated, info = env.step(random_action)
		print('reward',reward)