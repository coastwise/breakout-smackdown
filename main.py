import gym
env = gym.make('Breakout-v0')

import numpy
import time

import breakout_feature_engineering as features

# env.action_space allows for all 4 directions
# [up, down, right, left]
# i'm only going to allow two: [right, left]
rightleft_action_space = gym.spaces.Discrete(2)

print( env.observation_space )


for episode in range( 1 ):
	env.reset()

	observation, reward, done, info = env.step( 0 ) # no-op	
	observation, reward, done, info = env.step( 1 ) # need to start by pressing "down"
	t = 1

	play = features.extract_playarea_redchannel( observation )
	prev_play = play
	
	while not done:
		t += 1

		env.render()
		print( "tic" )

		delta = play - prev_play

		paddle_start, paddle_end = features.detect_paddle( play )
		ball_x, ball_y = features.detect_ball( delta, paddle_start, paddle_end )

		if ball_x != -1 and ball_y != -1:
			print( "ball", ball_x, ball_y )

			if ball_x < paddle_start:
				action = 3
			elif ball_x > paddle_end:
				action = 2
			else:
				action = 0

		else:
			print( "no ball!!" )
			action = 1 # down to serve

		#numpy.set_printoptions(threshold=numpy.nan)
		#print( numpy.argwhere( red == 142 ) )		

		prev_play = play
		observation, reward, done, info = env.step( action )
		play = features.extract_playarea_redchannel( observation )

		if done:
			print( "Episode finished after {} timesteps.", format( t+1 ) )
			break
