import numpy

paddle_min_x = 8
paddle_max_x = 151
paddle_min_y = 189
paddle_max_y = 192

paddle_colour = numpy.array([200,72,72])

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

	if paddle_start < paddle_end:
		print( "paddle between ", paddle_start, paddle_end )
	else:
		print( "no paddle found!" )

	return paddle_start, paddle_end
