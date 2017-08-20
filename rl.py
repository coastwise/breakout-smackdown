import numpy

def discounted_rewards( actual_rewards ):
	discount = 0.99
	result = numpy.zeros_like( actual_rewards )

	cumulative_reward = 0
	for t in xrange( actual_rewards.size - 1, 0, -1 ):
		cumulative_reward = cumulative_reward * discount + actual_rewards[ t ]
		result[ t ] = cumulative_reward

	return result

import tensorflow
import tensorflow.contrib.slim as slim

# what's relu? it's a "rectified linear unit"

state_dimensions = 3 # TODO
hidden_size = 18
action_size = 3
learning_rate = 0.01
nninput = tensorflow.placeholder( shape = [ None, state_dimensions ], dtype = tensorflow.float32 )
nnhidden = slim.fully_connected(  nninput, hidden_size, activation_fn = tensorflow.nn.relu,    biases_initializer = None )
nnoutput = slim.fully_connected( nnhidden, action_size, activation_fn = tensorflow.nn.softmax, biases_initializer = None )
nnaction = tensorflow.argmax( nnoutput, 1 )

reward_holder = tensorflow.placeholder( shape = [ None ], dtype = tensorflow.float32 )
action_holder = tensorflow.placeholder( shape = [ None ], dtype = tensorflow.int32 )

output_shape = tensorflow.shape( nnoutput )
indexes = tensorflow.range( 0, output_shape[ 0 ] )
indexes *= output_shape[ 1 ]
indexes += action_holder

responsible_outputs = tensorflow.gather( tensorflow.reshape( nnoutput, [-1] ), indexes )
loss = -tensorflow.reduce_mean( tensorflow.log( responsible_outputs ) * reward_holder )

tvars = tensorflow.trainable_variables()
gradient_holders = []
for i, var in enumerate( tvars ):
	placeholder = tensorflow.placeholder( tensorflow.float32, name = str(i)+'_holder' )
	gradient_holders.append( placeholder )

gradients = tensorflow.gradients( loss, tvars )
optimizer = tensorflow.train.AdamOptimizer( learning_rate = learning_rate )
update_batch = optimizer.apply_gradients( zip( gradient_holders, tvars ) )

###############################################

import gym
env = gym.make('Breakout-v0')

import breakout_feature_engineering as features

init = tensorflow.global_variables_initializer()
with tensorflow.Session() as session:
	session.run( init )

	episode_rewards = []
	episode_lengths = []

	grad_buffer = session.run( tensorflow.trainable_variables() )
	for i, grad in enumerate( grad_buffer ):
		grad_buffer[ i ] = grad * 0

	for episode in range( 10000 ):
		env.reset()

		observation, reward, done, info = env.step( 0 ) # no-op	
		prev_play = features.extract_playarea_redchannel( observation )

		observation, reward, done, info = env.step( 1 ) # need to start by pressing "down"
		play = features.extract_playarea_redchannel( observation )

		delta = play - prev_play
		state_i = features.ball_and_paddle_state( play, delta, [0,0,0] )
		state_f = features.normalize_state( state_i )

		t = 1
		total_reward = 0
		history = []
		
		while not done:
			t += 1

			a_dist = session.run( nnoutput, feed_dict = {nninput:[ state_f ]})
			action = numpy.random.choice( a_dist[0], p = a_dist[0] )
			action = numpy.argmax( a_dist == action )

			prev_play = play
			observation, reward, done, info = env.step( action + 1 )
			play = features.extract_playarea_redchannel( observation )
			delta = play - prev_play

			new_state_i = features.ball_and_paddle_state( play, delta, state_i )
			new_state_f = features.normalize_state( new_state_i )

			# custom reward & end condition
			reward = 1
			if new_state_f[1] < 0:
				# TODO: ball -1 happens even on good bounce sometimes... :/
				done = True

			history.append([ state_f, action, reward, new_state_f ])

			state_i = new_state_i
			state_f = new_state_f
			total_reward += reward

			if done:
				#print( "Episode finished after {} timesteps.", format( t+1 ) )
				#print( "total reward", total_reward )

				history = numpy.array( history )
				history[:,2] = discounted_rewards( history[:,2] )

				feed_dict = {
					reward_holder : history[:,2],
					action_holder : history[:,1],
					nninput : numpy.vstack( history[:,0] )
				}
				grads = session.run( gradients, feed_dict = feed_dict )
				for i, grad in enumerate( grads ):
					grad_buffer[i] += grad

				if i % 5 == 0 and i != 0:
					feed_dict = dict( zip( gradient_holders, grad_buffer ) )
					_ = session.run( update_batch, feed_dict = feed_dict )
					for i, grad in enumerate( grad_buffer ):
						grad_buffer[ i ] = grad * 0

				episode_rewards.append( total_reward )
				episode_lengths.append( t )

				#print( total_reward, t )

				break

		if episode % 100 == 0 and episode != 0:
			print( "running avg", numpy.mean( episode_rewards[-100:] ), numpy.mean( episode_lengths[-100:] ) )










