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
	paddle_start = paddle_max_x
	paddle_end = paddle_min_x
	y = 190
	
	for x in range( paddle_min_x, paddle_max_x + 1 ):
		pixel = observation[y][x]
		if pixel[0] == paddle_colour[0] and pixel[1] == paddle_colour[1] and pixel[2] == paddle_colour[2]:
			if x > paddle_end:
				paddle_end = x
			if x < paddle_start:
				paddle_start = x

	return paddle_start, paddle_end



play_min_x = 8
play_max_x = 151
play_min_y = 32
play_max_y = 204

ball_colour = numpy.array([200,72,72])

def detect_ball( observation, delta ):
	for y in range( play_min_y, play_max_y + 1 ):
		for x in range( play_min_x, play_max_x + 1 ):
			if delta[y][x][0] == ball_colour[0]:
				return x, y

	return 0, 0 # could not find

