import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from scipy import ndimage
from scipy.signal import medfilt
from scipy.stats import norm
from scipy import optimize
import progressbar
import os
import pickle
import pandas as pd

#returns array of frames and timestamps
def load_data(dir,img_type):
	"""
	Load in spitzer aor files by img type 
	Appropriate types are: bcd, pbcd, etc.
	Returns array of frame images and array of timestamps

	"""
	print ('Loading data...')
	frames = []
	times = []

	pbar = progressbar.ProgressBar()
	for i in pbar(os.listdir(dir)):
		if img_type + '.fits' in i:
			hdu = fits.open(dir + i)[0]

			data = hdu.data[:,:,:]
			data = np.nan_to_num(data)

			time = hdu.header['BMJD_OBS'] #timestamp for first frame
			dt = hdu.header['FRAMTIME'] #length of each frame in seconds 

			#since each fits file only has a timestamp and framedelay
			#manually calculate the timestap for each individual frame
			for j in range(len(data)):
				frames.append(np.asarray(data[j]))
				times.append(time + j*dt/3600.0/24.0)

	print ('Data loaded')
	print (' ')
	return frames, times

def median_filter(frames):
	"""
	median filter the pixels
	returns new array of filtered frames
	"""
	print ("Filtering data...")
	frames = np.asarray(frames)
	med_filtered_data = np.zeros(np.shape(frames))
			
	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			pixel_stream = frames[:,i,j]
			med_filtered_data[:,i,j] = medfilt(pixel_stream,kernel_size=5)

	diff = np.absolute(np.subtract(frames,med_filtered_data))

	#for each pixel, calculate std across all frames
	std = np.std(frames,axis = 0)

	count = 0
	pbar = progressbar.ProgressBar()
	for k in pbar(range(len(frames))):
		frame = frames[k]
		for i in range(len(frame)):
			for j in range(len(frame[0])):
				#if diff[k,i,j]/med_diff[k,i,j] > 4:
				if diff[k,i,j]/float(std[i,j]) > 4:
					#print (k,i,j)
					count += 1
					frames[k,i,j] = med_filtered_data[k,i,j]

	print ('Filtering done')
	filter_percent = str(round(float(count) / len(frames) / len(frames[0]) / len(frames[0][0]) * 100,3)) + ' %'
	print ('Fraction of pixels flagged: ' + filter_percent)
	print (' ')
	return frames

def get_background(frames,centroids,bckg_method):
	"""
	#calculate background using one of 2 methods
	#TODO: deal with cases where centroid returned -1 -1 (i.e. couldnt calculate 2dgaussian)
	
	Returns array of background values, one for each frame
	"""


	CORNER_SIZE = 10 #side length of square to calculate backgrounds from

	print ('Calculating background levels')
	frames = np.asarray(frames)
	backgrounds = []

	pbar = progressbar.ProgressBar()
	for i in pbar(range(len(frames))):
		if bckg_method == 'annulus':
			annulus_apertures = CircularAnnulus(centroids[i],r_in = 10., r_out = 15.)
			bkgflux = aperture_photometry(frames[i],annulus_apertures)
			bkg_mean = bkgflux['aperture_sum'] / annulus_apertures.area()
			backgrounds.append(bkg_mean[0])

		if bckg_method == 'corners':
			#calculate background by summing flux in corners of image, fit gaussian, get mean
			background = []
			background += list(np.reshape(frames[i,:CORNER_SIZE,:CORNER_SIZE],(CORNER_SIZE**2)))
			background += list(np.reshape(frames[i,-CORNER_SIZE:,:CORNER_SIZE],(CORNER_SIZE**2)))
			background += list(np.reshape(frames[i,:CORNER_SIZE,-CORNER_SIZE:],(CORNER_SIZE**2)))
			background += list(np.reshape(frames[i,-CORNER_SIZE:,-CORNER_SIZE:],(CORNER_SIZE**2)))

			mu, sigma = norm.fit(background)
			backgrounds.append(mu)

	print ('Backgrounds done calculating')
	print (' ')
	return backgrounds

def twoD_gaussian(x,amp,x0,y0,sigma_x,sigma_y):
	"""
	Function to calculate 2d gaussian, used for centroid
	"""
	return amp*np.exp(-(x[0]-x0)**2/sigma_x**2)*np.exp(-(x[1]-y0)**2/sigma_y**2)

def get_centroids(frames,centroid_method,path):
	"""
	Calculate centroids (x,y) for each frame
	2dgauss: get centroid by fitting 2dgaussian to center of image
	col: get centroid by measuring center of light (doesn't work if there's a companian)

	Returns [x,y] pair of coordinates for each frame in an array
	"""

	print ('Calculating centroids')
	centroids = []

	W = len(frames[0]) #Width
	H = len(frames[0][0]) #Height

	pbar = progressbar.ProgressBar()

	#fit 2d gaussian to get pos of star, assume in middle of ccd to start
	gauss_guess = (np.max(frames[0]),W/2.0,H/2.0,1,1)

	for i in pbar(range(len(frames))):
		frame = np.nan_to_num(frames[i])
		#get centroid by fitting 2dgaussian to center of image
		if centroid_method == '2dgauss':
			#get flux from aperture photometry
			X, Y = np.meshgrid(range(W),range(H))
			xdata = np.vstack((X.reshape((1,W*H)), Y.reshape((1,W*H))))

			try:
				popt,pcov = optimize.curve_fit(twoD_gaussian,xdata,frame.reshape(W*H),p0 = gauss_guess)
				gauss_guess = popt #use prev result to guess next one
				
				positions = [popt[1],popt[2]] #get pos of star on ccd
			except: #if for some reason it can't find the centroid
				print ('Error calculating 2dgaussian centroid on frame: ' + str(i))
				plt.imshow(np.log10(abs(frame)))
				plt.colorbar()
				plt.savefig(path + 'bad_gaussian_centroid_' + str(i) + '.png')
				plt.clf()
				positions = [-1,-1]


		#center of light
		if centroid_method == 'col':
			positions = np.asarray(ndimage.measurements.center_of_mass(frame))

		centroids.append(positions)


	print ('Centroids calculated')
	print (' ')
	return centroids

def subtract_background(frames,bckg):
	'''
	subtract the frame background value from each pixel in frame
	return subtracted frames
	'''
	return [np.subtract(frames[i],bckg[i]) for i in range(len(frames))]


def make_analysis_plots(times,centroids,bckg,path):
	"""
	makes diagnostic plots about centroids/backgrounds
	"""


	#x/ycentroid over time
	plt.scatter(times,[zz[0] for zz in centroids],s = 2,label = 'x')
	plt.scatter(times,[zz[1] for zz in centroids],s = 2, label = 'y')
	plt.xlabel('Time (BMJD)')
	plt.ylabel('Centroid')
	plt.legend(loc = 0)
	plt.savefig(path + 'centroid_timeseries.pdf')
	plt.clf()
	
	#x vs y centroid
	plt.scatter([i[0] for i in centroids],[i[1] for i in centroids],s = 2)
	plt.xlabel('x centroid')
	plt.ylabel('y centroid')
	plt.savefig(path + 'centroid_position.pdf')
	plt.clf()

	#background over time
	plt.scatter(times,bckg,s = 2,c = 'C1')
	plt.savefig(path + 'background_timeseries.pdf')
	plt.clf()

	#histogram of background values
	plt.hist(bckg,bins = 30)
	plt.xlabel('Background value')
	plt.savefig(path + 'background_histogram.pdf')
	plt.clf()
	
	return

def median_stack(frames):
	"""
	median stack all the frames
	returns single frame image
	"""
	return np.median(frames,axis = 0)

def check_if_storage_dir(path):
	"""
	check if directory to store stuff exists, if not then make it
	"""
	if not os.path.exists(path):
		os.makedirs(path)

def save_frame_data(times,frames,median_frame,centroids,bckg,path):
	"""
	Save data regarding corrected frames
	"""

	#get rid of nans before saving
	frames = np.nan_to_num(frames)
	median_frame = np.nan_to_num(median_frame)

	#save corrected frames and times
		#pickle.dump([times,frames],open(path + 'median_filtered_bckg_subtracted_frames_times.p','wb'))
	np.save(path + 'median_filtered_bckg_subtracted_frames',frames)
	np.save(path + 'times',times)

	#save centroids,background
	d = pd.DataFrame()
	d['x_centroid'] = [i[0] for i in centroids]
	d['y_centroid'] = [i[1] for i in centroids]
	d['backgrounds'] = bckg
	d.to_csv(path + 'centroid_bckg_data.csv',index=False)

	#plot of median_stack
	plt.imshow(np.log10(np.abs(median_frame)))
	plt.colorbar(label = r'$log_{10} \mathrm{(flux)}$')
	plt.savefig(path + 'median_stacked_img.pdf')

	#pickle.dump(median_frame,open(path + 'median_stacked_frame.p','wb'))
	np.save(path + 'median_stacked_frame',median_frame)
	#flatten the median stack
	median_stack = np.abs(np.reshape(median_frame,(len(median_frame)*len(median_frame[0]))))

	#get order of brightest pixels
	order = np.asarray(sorted(range(len(median_stack)),key=lambda k: median_stack[k])[::-1])
	#sort median stack from high to low
	median_stack.sort()
	median_stack = np.asarray(median_stack[::-1])


	#generate list of pixel, relative flux it containts, and cumulative flux
	d = pd.DataFrame()
	d['pixel x'] = order % len(median_frame[0])
	d['pixel y'] = [int(i / len(median_frame)) for i in order]
	total = np.sum(median_stack)
	d['percent of flux'] = [round(i / total * 100,2) for i in median_stack]
	d['cumulative flux percentage'] = [round(i / total * 100,2) for i in np.cumsum(median_stack)]
	d.to_csv(path + 'pixel_flux_distribution.csv',index=False)


def default_run(data_path,CENTROID_METHOD,BCKG_METHOD):
	"""
	this is a default set up that will, given path to data and background
	 and centroid methods, calculate all of the standard stuff
	"""

	#make records of stuff
	path = './fits_analysis_data_' + CENTROID_METHOD + '_' + BCKG_METHOD + '/'
	#check if analysis directory exists, ifnot make it
	check_if_storage_dir(path)

	#load data
	frames, times = load_data(data_path,'bcd')

	#pass frames through a median filter
	frames = median_filter(frames)

	#calculate centroid
	centroids = get_centroids(frames,CENTROID_METHOD,path)

	#get rid of bad frames with bad centroid
	times = [times[i] for i in range(len(times)) if not(centroids[i][0] == -1)]
	frames = [frames[i] for i in range(len(frames)) if not(centroids[i][0] == -1)]
	centroids = [centroids[i] for i in range(len(centroids)) if not(centroids[i][0] == -1)]

	#calculate background
	bckg = get_background(frames,centroids,bckg_method=BCKG_METHOD)

	#subtract background from frames
	frames = subtract_background(frames,bckg)

	#get a median stack of the frames
	median_frame = median_stack(frames)
	
	#diagonistic plots about background and centroids
	make_analysis_plots(times,centroids=centroids,bckg=bckg,path=path)

	#save data
	save_frame_data(times,frames,median_frame,centroids,bckg,path)










