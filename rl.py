import getopt, sys
try:
	opts, args = getopt.getopt( sys.argv[1:], "", ["model=", "episodes=", "justplay", "headless"])
except getopt.GetoptError as err:
	print str( err )
	sys.exit( 2 )


model_file = ''
max_episodes = 1000
headless = False
just_play = False


for opt, arg in opts:
	if opt == "--model":
		model_file = arg
	if opt == "--episodes":
		max_episodes = int( arg )
		print 'max_episodes = {}'.format(max_episodes)
	if opt == "--justplay":
		just_play = True
	if opt == "--headless":
		headless = True


import numpy

def discounted_rewards( actual_rewards ):
	discount = 0.8
	result = numpy.zeros_like( actual_rewards )

	cumulative_reward = 0
	for t in xrange( actual_rewards.size - 1, 0, -1 ):
		cumulative_reward = cumulative_reward * discount + actual_rewards[ t ]
		result[ t ] = cumulative_reward

	return result

###############################################################################

import tensorflow
import tensorflow.contrib.slim as slim

state_dimensions = 5
hidden_size = 18
action_size = 3
learning_rate = 0.01

nninput = tensorflow.placeholder( shape = [ None, state_dimensions ], dtype = tensorflow.float32 )
nnhidden = slim.fully_connected(  nninput, hidden_size, activation_fn = tensorflow.nn.relu,    biases_initializer = None )
nnoutput = slim.fully_connected( nnhidden, action_size, activation_fn = tensorflow.nn.softmax, biases_initializer = None )

reward_holder = tensorflow.placeholder( shape = [ None ], dtype = tensorflow.float32 )
action_holder = tensorflow.placeholder( shape = [ None, 3 ], dtype = tensorflow.float32 )

action_readout = tensorflow.reduce_sum( nnoutput * action_holder, axis = 1 )
loss = tensorflow.reduce_mean( tensorflow.square( reward_holder - action_readout ) )
optimize = tensorflow.train.AdamOptimizer( learning_rate ).minimize( loss )

###############################################################################

import gym
env = gym.make('Breakout-v0')

import breakout_feature_engineering as features

saver = tensorflow.train.Saver()
init = tensorflow.global_variables_initializer()
with tensorflow.Session() as session:
	session.run( init )

	if model_file != '':
		print 'restoring {}'.format( model_file )
		try:
			saver.restore( sess = session, save_path = model_file )
		except:
			print 'failed'


	episode_rewards = []
	episode_lengths = []

	for episode in range( max_episodes ):
		env.reset()

		observation, reward, done, info = env.step( 0 ) # no-op
		prev_play = features.extract_playarea_redchannel( observation )

		observation, reward, done, info = env.step( 1 ) # need to start by pressing "down"
		play = features.extract_playarea_redchannel( observation )

		delta = play - prev_play
		state_i = features.ball_and_paddle_state( play, delta, [0,0,0,0,0] )
		state_f = features.normalize_state( state_i )

		if not headless:
			env.render()

		t = 1
		total_reward = 0
		history = []

		while not done:
			t += 1

			# choose next action
			a_dist = session.run( nnoutput, feed_dict = {nninput:[ state_f ]})[0]
			#action = numpy.random.choice( action_size, p = a_dist ) # pick random weighted by how good we think it is
			action = numpy.argmax( a_dist ) # pick best

			prev_play = play
			observation, reward, done, info = env.step( action + 1 )
			play = features.extract_playarea_redchannel( observation )
			delta = play - prev_play

			new_state_i = features.ball_and_paddle_state( play, delta, state_i )
			new_state_f = features.normalize_state( new_state_i )

			# custom reward
			if state_i[4] < 0 and new_state_i[4] >= 0:
				# +1 for bounce
				reward = 1

			# custom end condition
			if new_state_i[1] < 0:
				# drop the ball? punish & game over
				reward = -1
				done = True

			actions = [0,0,0]
			actions[action] = 1
			history.append([ state_f, actions, reward, new_state_f ])

			state_i = new_state_i
			state_f = new_state_f
			total_reward += reward

			if not headless:
				env.render()

			if done:
				#print( "Episode finished after {} timesteps.", format( t+1 ) )
				#print( "total reward", total_reward )

				history = numpy.array( history )

				history[:,2] = discounted_rewards( history[:,2] )

				feed_dict = {
					reward_holder : history[:,2],
					action_holder : numpy.vstack( history[:,1] ),
					nninput : numpy.vstack( history[:,0] )
				}

				session.run( optimize, feed_dict = feed_dict )

				episode_rewards.append( total_reward )
				episode_lengths.append( t )

				#print( total_reward, t )

				break

		if episode % 100 == 0 and episode != 0:
			print( "running avg", numpy.mean( episode_rewards[-100:] ), numpy.mean( episode_lengths[-100:] ) )

	if model_file != '':
		print 'saving {}'.format( model_file )
		saver.save( sess = session, save_path = model_file)







