from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry


#calculate flux and save it to a file
def calc_flux_radius(path,frames,centroids,r):
	output = open(path + 'lightcurves/r_' + str(r) + '_flux.data','w+')

	for i in range(len(frames)):
		star_aperture = CircularAperture(centroids[i], r=r)
		rawflux = aperture_photometry(frames[i], star_aperture)
		flux = rawflux['aperture_sum']
		output.write(str(flux[0]) + '\n')
	return

#TODO
def calc_flux_variable(path,frames,centroids):
	return

#TODO
def calc_flux_exact(path,frames,centroids,pixels):
	return

