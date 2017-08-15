import gym
env = gym.make('Breakout-v0')

import numpy

# env.action_space allows for all 4 directions
# [up, down, right, left]
# i'm only going to allow two: [right, left]
rightleft_action_space = gym.spaces.Discrete(2)

print( env.observation_space )

paddle_min_x = 8
paddle_max_x = 151
paddle_min_y = 189
paddle_max_y = 192

for episode in range( 1 ):
	prev_obs = env.reset().astype( numpy.int16 )
	#obs, reward, done, info = env.step( 1 ) # need to start by pressing "down"
	#obs = obs.astype( numpy.int16 )
	obs = prev_obs

	for t in range( 5 ):
		#env.render()
		print( "tic" )

		delta = obs - prev_obs

		'''for col_idx, row in enumerate( delta ):
			for row_idx, pixel in enumerate( row ):
				if sum( pixel ) != 0:
					print( row_idx, col_idx, pixel )
					print( obs[col_idx][row_idx] )
		'''

		paddle_start = paddle_max_x
		paddle_end = paddle_min_x
		y = 190
		needle = numpy.array([200,72,72])
		for x in range( paddle_min_x, paddle_max_x + 1 ):
			pixel = obs[y][x]
			if pixel[0] == needle[0] and pixel[1] == needle[1] and pixel[2] == needle[2]:
				if x > paddle_end:
					paddle_end = x
				if x < paddle_start:
					paddle_start = x

		if paddle_start < paddle_end:
			print( "paddle between ", paddle_start, paddle_end )
		else:
			print( "no paddle found!" )

		rightleft_action = rightleft_action_space.sample()
		action = rightleft_action + 2 # convert Discrete(2) to last two elements of Discrete(4)

		action = 2 # move paddle right

		prev_obs = obs #.astype(int16)
		obs, reward, done, info = env.step( action )
		obs = obs.astype( numpy.int16 )

		if done:
			print( "Episode finished after {} timesteps.", format( t+1 ) )
			break
