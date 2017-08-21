import numpy

paddle_min_x = 8
paddle_max_x = 151
paddle_min_y = 189
paddle_max_y = 192

paddle_colour = numpy.array([200,72,72])
ball_colour = numpy.array([200,72,72])
block_colours = numpy.array([
                              [ 66,  72, 200],
                              [ 72, 160,  72],
                              [162, 162,  42],
                              [180, 122,  48],
                              [198, 108,  58],
                              [200,  72,  72]
                            ])
border_colour = [142, 142, 142]
left_of_paddle_colour = [66, 158, 130]

def detect_paddle( observation ):
	paddle_start = paddle_max_x + 1
	paddle_end = paddle_min_x - 1

	paddle_top = observation[157, :]
	paddle = numpy.where( paddle_top == paddle_colour[0] )

	paddle_start = paddle[0][0]
	paddle_end = paddle[0][-1]

	if paddle_end - paddle_start > 16:
		paddle_bottom = observation[160, :]
		paddle = numpy.where( paddle_bottom == paddle_colour[0] )
		paddle_start = paddle[0][0]
		paddle_end = paddle[0][-1]

	return paddle_start, paddle_end


play_min_x = 8
play_max_x = 151
play_min_y = 32
play_max_y = 204

def extract_playarea_redchannel( observation ):
	return observation[ play_min_y : play_max_y, play_min_x : play_max_x, 0 ].astype( numpy.int16 )


ball_colour = numpy.array([200,72,72])

def detect_ball( obs, delta, paddle_start, paddle_end ):
	red_pixels_y, red_pixels_x = numpy.where( obs == ball_colour[0] )

	for i in range( len( red_pixels_y ) ):
		y = red_pixels_y[i]
		if y >= 25 and y <= 30:
			continue # this is overzealous, but only if the agent can break into the back rows...

		if y >= 157 and y <= 160:
			if red_pixels_x[i] >= paddle_start and red_pixels_x[i] <= paddle_end:
				continue
		return red_pixels_x[i], red_pixels_y[i]

	return -1, -1 # not found

def ball_and_paddle_state( observation, delta, prev_state ):
	paddle_start, paddle_end = detect_paddle( observation )
	ball_x, ball_y = detect_ball( observation, delta, paddle_start, paddle_end )

	prev_ball_x = prev_state[ 1 ]
	prev_ball_y = prev_state[ 2 ]

	delta_ball_x = prev_ball_x - ball_x
	delta_ball_y = prev_ball_y - ball_y
	
	if ball_x < 0:
		if prev_ball_y < 150:
			# ignore missing ball if it wasn't near the bottom of the screen
			# it probably changed colours due to atari hardware limits
			# TODO: more robust ball detection
			ball_x = prev_ball_x
			ball_y = prev_ball_y
		# no ball, no dela ball
		delta_ball_x = 0
		delta_ball_y = 0			

	return [ paddle_start, ball_x, ball_y, delta_ball_x, delta_ball_y ]


play_width = play_max_x - play_min_x
play_height = play_max_y - play_min_y

def normalize_state( int_state ):

	paddle_range = 136

	return [
		-1 + 2 * float( int_state[0] ) / paddle_range,
		float( int_state[1] ) / play_width,
		float( int_state[2] ) / play_height,
		float( int_state[3] ) / 10,
		float( int_state[4] ) / 10
	]
