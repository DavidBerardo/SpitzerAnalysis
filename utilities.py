from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

def import_saved_data(path):
	"""
	convenience function to load all the saved files 
	"""
	cent_back = pd.read_csv(path + 'centroid_bckg_data.csv')
	centroids = [[cent_back['x_centroid'][i],cent_back['y_centroid'][i]] for i in range(len(cent_back))]
	bckg = cent_back['backgrounds']

	times = np.load(path + 'times.npy')
	frames = np.load(path + 'median_filtered_bckg_subtracted_frames.npy')
	#times, frames = pickle.load(open(path + 'median_filtered_bckg_subtracted_frames_times.p','rb'))
	
	#median_frame = pickle.load(open(path + 'median_stacked_frame.p','rb'))
	median_frame = np.load(path + 'median_stacked_frame.npy')
	return times,frames,centroids,bckg,median_frame

def import_specific(path,data_names):
	"""
	import only specific pieces of data
	"""

	outputs = []
	for i in data_names:
		if i == 'centroids':
			if not('cent_back' in vars()): #check if already loaded
				cent_back = pd.read_csv(path + 'centroid_bckg_data.csv')
			
			centroids = np.asarray([[cent_back['x_centroid'][i],cent_back['y_centroid'][i]] for i in range(len(cent_back))])
			outputs.append(centroids)

		if i == 'bckg':
			if not('cent_back' in vars()): #check if already loaded
				cent_back = pd.read_csv(path + 'centroid_bckg_data.csv')
	
			bckg = cent_back['backgrounds']
			outputs.append(bckg)


		if i == 'times':
			#if not('times' in vars()): #check if already loaded
			#	times, frames = pickle.load(open(path + 'median_filtered_bckg_subtracted_frames_times.p','rb'))
			times = np.load(path + 'times.npy')
			outputs.append(times)

		if i == 'frames':
			#check if we already loaded the data
			#if not('frames' in vars()): #check if already loaded
			#	times, frames = pickle.load(open(path + 'median_filtered_bckg_subtracted_frames_times.p','rb'))
			frames = np.load(path + 'median_filtered_bckg_subtracted_frames.npy')
			outputs.append(frames)

		if i == 'median_frame':
			#median_frame = pickle.load(open(path + 'median_stacked_frame.p','rb'))
			median_frame = np.load(path + 'median_stacked_frame.npy')
			outputs.append(median_frame)

		if i == 'pixel_dist':
			pixel_dist = pd.read_csv(path + 'pixel_flux_distribution.csv')

			outputs.append(pixel_dist)

	#without this, returns object in an array if theres only one thing
	if len(outputs) == 1:
		return outputs[0]

	return outputs

def calc_centroid_outliers(path,sigma):
	centroids = import_specific(path,['centroids'])
	med_x, med_y = np.median(centroids,axis=0)
	sigma_x,sigma_y = np.std(centroids,axis=0)

	mask = [False if (abs(i[0] - med_x)/sigma_x < sigma and abs(i[1] - med_y)/sigma_y < sigma) else True for i in centroids]
	np.save(path + 'centroid_mask_' + str(sigma)+ '_sigma',mask)
	return 

#TODO load flux with appropriate masks
def load_flux_time(path,flux,cent_mask = None):
	times = import_specific(path,['times'])
	flux = open(path + 'lightcurves/' + flux,'r').readlines()
	flux = [float(i.strip()) for i in flux]
	flux = flux / np.nanmedian(flux)
	try:
		times = np.asarray([times[i] for i in range(len(times)) if ~cent_mask[i]])
		flux = np.asarray([flux[i] for i in range(len(flux)) if ~cent_mask[i]])
	except:
		pass


	return times,flux

#load flux and also normalize
def load_flux(path,radius):
	flux = open(path + 'lightcurves/r_' + str(radius) + '_flux.data','r').readlines()
	flux = np.asarray([float(i) for i in flux])
	flux = flux / np.nanmedian(flux)
	return flux

def load_pld_pixels(path,pixel_file,cent_mask = None):
	#don't need first line
	data = open(path + pixel_file).readlines()[1:]

	try:
		cent_mask = np.load(path + cent_mask)
		data = np.asarray([data[i] for i in range(len(data)) if ~cent_mask[i]])
	except:
		pass

	#get number of pld pixels
	n = len(data[0].strip().split(','))
	pixels = []
	for i in range(n):
		pixels.append(np.asarray([float(j.strip().split(',')[i]) for j in data]))

	return pixels


