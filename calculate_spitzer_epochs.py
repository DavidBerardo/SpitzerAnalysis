from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def get_epochs(start_time,end_time,t0,p):
	n1 = (start_time - t0) / p
	n2 = (end_time - t0) / p

	outline = 'n1: ' + str(n1) + ', n2: ' + str(n2)
	if int(n2) - int(n1) > 0:
		outline += ', yes'
	else:
		outline += ', no'

	print (outline)
	return n1, n2