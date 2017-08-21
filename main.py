import numpy
import gym
env = gym.make('Breakout-v0')

import breakout_feature_engineering as features

# env.action_space allows for all 4 directions
# [up, down, right, left]
# i'm only going to allow two: [right, left]
rightleft_action_space = gym.spaces.Discrete(2)

for episode in range( 4 ):
	env.reset()

	observation, reward, done, info = env.step( 0 ) # no-op	
	prev_play = features.extract_playarea_redchannel( observation )

	observation, reward, done, info = env.step( 1 ) # need to start by pressing "down"
	play = features.extract_playarea_redchannel( observation )
	delta = play - prev_play
	state = features.ball_and_paddle_state( play, delta, [0,0,0,0,0] )

	t = 1
	total_reward = 0

	while not done:
		t += 1

		#env.render()

		ball_x, ball_y = state[1], state[2]
		paddle_start, paddle_end = state[0], state[0]+16

		if ball_x != -1 and ball_y != -1:

			if ball_x < paddle_start:
				action = 3
			elif ball_x > paddle_end:
				action = 2
			else:
				action = 0

		else:
			action = 1 # down to serve


		observation, reward, done, info = env.step( action )
		new_play = features.extract_playarea_redchannel( observation )
		delta = new_play - play

		new_state = features.ball_and_paddle_state( new_play, delta, state )

		if new_state[1] < 0:
			# TODO: ball -1 happens even on good bounce sometimes... :/
			done = True

		play = new_play
		state = new_state
		total_reward += reward

		if done:
			print( "Episode finished after {} timesteps.", format( t+1 ) )
			print( "total reward", total_reward )
			break
