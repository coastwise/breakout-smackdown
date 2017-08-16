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

	print( "paddle", paddle_start, paddle_end )

	if paddle_end - paddle_start > 16:
		print( "paddle too long!")
		paddle_bottom = observation[160, :]
		paddle = numpy.where( paddle_bottom == paddle_colour[0] )
		paddle_start = paddle[0][0]
		paddle_end = paddle[0][-1]
		print( "paddle", paddle_start, paddle_end )

	return paddle_start, paddle_end



def extract_playarea_redchannel( observation ):
	play_min_x = 8
	play_max_x = 151
	play_min_y = 32
	play_max_y = 204
	return observation[ play_min_y : play_max_y, play_min_x : play_max_x, 0 ].astype( numpy.int16 )


ball_colour = numpy.array([200,72,72])

def detect_ball( delta, paddle_start, paddle_end ):
	for y, row in enumerate( delta ):
		for x, red_value in enumerate( row ):
			if red_value == ball_colour[0]:
				return x, y

	return -1, -1 # not found


