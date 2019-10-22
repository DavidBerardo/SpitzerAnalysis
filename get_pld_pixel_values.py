from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from utilities import import_specific
import pandas as pd

"""
Normalize the frames, and extract the pixel values to be used in pld fitting
Need processed frames, as well as list of pixels to extract
"""

def n_by_n(frames,centroids,n,path):
	"""
	gets pld pixels from nxn square around centroid
	"""
	x = int(round(np.mean([i[0] for i in centroids]))) #get x pixel coord
	y = int(round(np.mean([i[1] for i in centroids]))) #get y pixel coord

	#output to store data
	output1 = open(path + str(n) + 'x' + str(n) + '_pld_pixels.data','w+')

	#generate string of pixels
	pix_str = '(x,y): '
	for i in range(y - int(n/2),y + int(n/2) + 1):
			for j in range(x - int(n/2),x +int(n/2) + 1):
				pix_str += '(' + str(j) + ',' + str(i) + ') '

	output1.write(pix_str + '\n')

	#output to store the flux to compare with other lightcurves
	output2 = open(path + 'lightcurves/' + str(n) + 'x' + str(n) + '_flux.data','w+')

	#loop over frame, get normalized pixel values, write to file
	for f in frames:
		pixel_fluxes = []
		for i in range(y - int(n/2),y + int(n/2) + 1):
			for j in range(x - int(n/2),x +int(n/2) + 1):
				pixel_fluxes.append(f[i][j])

		pixel_fluxes_sum = np.sum(pixel_fluxes)
		pixel_fluxes = [i/pixel_fluxes_sum for i in pixel_fluxes]

		output1.write(','.join([str(z) for z in pixel_fluxes]) + '\n')
		output2.write(str(pixel_fluxes_sum) + '\n')
	return

#gets pld_pixles from pixels contributing
#a certain fraction of flux to total frame
def top_pixels(frames,n,path):
	"""
	Take top n pixels by flux contribution in median image
	"""

	#output to store data
	output1 = open(path + 'top_' + str(n) + '_flux_pld_pixels.data','w+')


	#load in distribution of pixels	
	dist = import_specific(path,['pixel_dist'])[:n]
	x_list = [int(dist.iloc[i]['pixel x']) for i in range(n)]
	y_list = [int(dist.iloc[i]['pixel y']) for i in range(n)]

	#generate string of pixels
	pix_str = '(x,y): '
	for i in range(n):
		pix_str += '(' + str(x_list[i]) + ',' + str(y_list[i]) + ') '
	output1.write(pix_str + '\n')

	#output to store the flux to compare with other lightcurves
	output2 = open(path + 'lightcurves/top_' + str(n) + '_flux.data','w+')

	#loop over frame, get normalized pixel values, write to file
	for f in frames:
		pixel_fluxes = []
		for i in range(n):
			pixel_fluxes.append(f[y_list[i]][x_list[i]])

		pixel_fluxes_sum = np.sum(pixel_fluxes)
		pixel_fluxes = [i/pixel_fluxes_sum for i in pixel_fluxes]
	
		output1.write(','.join([str(z) for z in pixel_fluxes]) + '\n')
		output2.write(str(pixel_fluxes_sum) + '\n')
	return

#TODO set this up to work with any generic list of pixels
def custom_flux(pixel_list):

	return