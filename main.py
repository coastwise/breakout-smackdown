import gym
env = gym.make('Breakout-v0')

# env.action_space allows for all 4 directions
# [up, down, right, left]
# i'm only going to allow two: [right, left]
rightleft_action_space = gym.spaces.Discrete(2)

print( env.observation_space )

for episode in range(4):
	prev_obs = env.reset()
	observation, reward, done, info = env.step( 1 ) # need to start by pressing "down"

	for t in range(100):
		env.render()

		rightleft_action = rightleft_action_space.sample()
		action = rightleft_action + 2 # convert Discrete(2) to last two elements of Discrete(4)
		observation, reward, done, info = env.step( action )

		if done:
			print( "Episode finished after {} timesteps.", format( t+1 ) )
			break
