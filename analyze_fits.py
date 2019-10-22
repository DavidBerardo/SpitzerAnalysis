import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm, pyplot as plt
from astropy.io import fits
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from scipy import ndimage
from scipy.signal import medfilt
from scipy.stats import norm
from scipy import optimize
from astropy.stats import sigma_clip
import progressbar
import time
import os

def load_data(aor,img_type):
	print 'Loading data...'
	frames = []
	times = []
	dir = 'r' + str(aor) + '/ch2/bcd/'
	#number of bcd fits files
	N = len([zz for zz in os.listdir(dir) if img_type + '.fits' in zz])
	pbar = progressbar.ProgressBar()
	for i in pbar(range(N)):
		fname = dir + 'SPITZER_I2_' + str(aor) + '_' + '0'*(4-len(str(i))) + str(i) + '_0000_1_' + img_type + '.fits'
		hdu = fits.open(fname)[0]

		data = hdu.data[:,:,:]
		data = np.nan_to_num(data)

		time = hdu.header['BMJD_OBS'] #timestap for first frame
		dt = hdu.header['FRAMTIME'] #length of each frame in seconds 

		for j in range(len(data)):
			frames.append(np.asarray(data[j]))
			times.append(time + j*dt/3600.0/24.0)
		
		#frames.append(data)
		#times.append(hdu.header['BMJD_OBS'])
	print 'Data loaded'
	print ' '
	return frames, times

#median filter the pixels
def median_filter(frames):
	print "Filtering data..."
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

	print 'Filtering done'
	print 'Fraction of pixels flagged: ' + str(round(float(count) / len(frames) / len(frames[0]) / len(frames[0][0]) * 100,3)) + ' %'
	print ' '
	return frames


def get_background(frames,centroids,bckg_method):
	print 'Calculating background levels'
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
			N = 10
			background += list(np.reshape(frames[i,:N,:N],(N**2)))
			background += list(np.reshape(frames[i,-N:,:N],(N**2)))
			background += list(np.reshape(frames[i,:N,-N:],(N**2)))
			background += list(np.reshape(frames[i,-N:,-N:],(N**2)))

			mu, sigma = norm.fit(background)
			backgrounds.append(mu)

	print 'Backgrounds done calculating'
	print ' '
	return backgrounds

def twoD_gaussian(x,amp,x0,y0,sigma_x,sigma_y):
	return amp*np.exp(-(x[0]-x0)**2/sigma_x**2)*np.exp(-(x[1]-y0)**2/sigma_y**2)

def get_centroids(frames,centroid_method = '2dgauss'):
	print 'Calculating centroids'
	centroids = []

	pbar = progressbar.ProgressBar()
	for i in pbar(range(len(frames))):
		frame = np.nan_to_num(frames[i])
		#get centroid by fitting 2dgaussian to center of image
		if centroid_method == '2dgauss':
			#get flux from aperture photometry
			X, Y = np.meshgrid(range(32),range(32))
			xdata = np.vstack((X.reshape((1,32*32)), Y.reshape((1,32*32))))

			#fit 2d gaussian to get pos of star, assume in middle of ccd to start
			initial_guess = (100,16,16,1,1)
			try:
				popt,pcov = optimize.curve_fit(twoD_gaussian,xdata,frame.reshape(32*32),p0 = initial_guess)
			except:
				'Error calculating 2dgaussian centroid on frame: ' + str(i)
				plt.imshow(frame)
				plt.show()
				sys.exit()	

			#get pos of star on ccd
			positions = [popt[1],popt[2]]


		#center of light
		if centroid_method == 'col':
			positions = np.asarray(ndimage.measurements.center_of_mass(frame))

		centroids.append(positions)


	print 'Centroids calculated'
	print ' '
	return centroids


def get_flux(frame,background,centroids,r,n,x,y):
	data = np.nan_to_num(frame)

	pixel_fluxes = []
	for i in range(y - n/2,y + n/2 + 1):
		for j in range(x - n/2,x + n/2 + 1):
			pixel_fluxes.append(data[i][j] - background)

	pixel_fluxes_sum = np.sum(pixel_fluxes)
	pixel_fluxes = [i/pixel_fluxes_sum for i in pixel_fluxes]

	#calculate flux as sum of discrete pixels
	if r == 'pld':
		flux = pixel_fluxes_sum
		return [flux] + pixel_fluxes

	#calculate flux using a variable radius
	if r == 'var':
		#calculate beta, noise pixel parameter from 
		#http://iopscience.iop.org/article/10.1088/0004-637X/766/2/95/pdf
		b = np.sum(frame)**2 / (np.sum(frame**2))
		star_aperture = CircularAperture(centroids, r=b**0.5)

	else:
		star_aperture = CircularAperture(centroids, r=r)

	rawflux = aperture_photometry(data, star_aperture)

	#how much to remove from star aperture
	bkg_sum = background * star_aperture.area()

	flux = rawflux['aperture_sum'] - bkg_sum
	return [flux[0]] + pixel_fluxes

#return mask for sigma clipping
def sig_clip_mask(flux_data):
	fluxes = [i[1] for i in flux_data]
	clipped = sigma_clip(list(fluxes),sigma = 4)
	return clipped.mask

def centroid_mask(centroids):
	cx, cy = [i[0] for i in centroids], [i[1] for i in centroids]
	mux, muy = np.mean(cx), np.mean(cy)
	sx, sy = np.std(cx), np.std(cy)

	mask = [False if (np.abs(i[0] - mux)/sx < 4) and (np.abs(i[1] - muy)/sy < 4) else True for i in centroids]
	return mask

def make_analysis_plots(times,centroid_method,cent_mask,bckg_method,centroids,backgrounds):
	fig = plt.figure(figsize = (15,15))
	N = len(times)

	#x/ycentroid over time
	ax = fig.add_subplot(2,2,1)
	plt.scatter(times,[zz[0] for zz in centroids],s = 2,label = 'x')
	plt.scatter(times,[zz[1] for zz in centroids],s = 2, label = 'y')
	plt.xlabel('time')
	plt.ylabel('centroid')
	plt.legend(loc = 0)
	
	#x vs y centroid
	ax = fig.add_subplot(2,2,2)
	plt.scatter([i[0] for i in centroids],[i[1] for i in centroids],s = 2)
	plt.scatter([centroids[i][0] for i in range(N) if cent_mask[i]],[centroids[i][1] for i in range(N) if cent_mask[i]],s = 2,c = 'C3',label = 'masked')
	plt.xlabel('x centroid')
	plt.ylabel('y centroid')
	plt.legend(loc = 0)

	#background over time
	ax = fig.add_subplot(2,2,3)
	plt.scatter(times,backgrounds,s = 2,c = 'C1')

	#histogram of background values
	ax = fig.add_subplot(2,2,4)
	plt.hist(backgrounds,bins = 30)
	plt.xlabel('Background value')
	plt.tight_layout()

	plt.savefig('fits_analysis_data/' + centroid_method + '_' + bckg_method + '.png')
	return

#get a timeseries of flux for different aperture radii
def get_lightcurve(centroid_method,bckg_method):
	aor = 62714368
	frames, times = load_data(aor,img_type='bcd') #load data
	print len(frames)
	frames = median_filter(frames) #filter bad pixels


	centroids = get_centroids(frames,centroid_method = centroid_method) #get the center of psf
	x = int(round(np.mean([i[0] for i in centroids]))) #get center for pld pixels
	y = int(round(np.mean([i[1] for i in centroids]))) #get center for pld pixels
	

	#find points to ignore based on anomalous centroids
	cent_mask = centroid_mask(centroids)
	print len([i for i in cent_mask if i])

	backgrounds = get_background(frames,centroids,bckg_method) #estimate background flux

	make_analysis_plots(times,centroid_method,cent_mask,bckg_method,centroids,backgrounds)

	radii = [1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,'pld']
	pld_square_size = [3]
	fig = plt.figure()
	offset = 0
	for r in radii:
		print 'Aperture radius: ' + str(r)
		for n in pld_square_size:
			print '    Square size: ' + str(n)
			
			output = open('fits_analysis_data/' + str(n) + 'x' + str(n) + '_photometry_radius_' + str(r) + '_' + centroid_method + '.data','w')
			output.write('time,flux,' + ','.join(['P' + str(i+1) for i in range(n**2)])+ ',x_cent,y_cent,bckg,cent_flag,sigma_clip_flag\n')

			pbar = progressbar.ProgressBar()
			all_data = []
			for j in pbar(range(len(frames))):
				#compute flux using given radius and centroid
				flux = get_flux(frames[j],backgrounds[j],centroids[j],r,n,x,y)
				all_data.append([times[j]] + flux + [centroids[j][0],centroids[j][1],backgrounds[j]])
			#get mask for sigma clipping
			clip_mask = sig_clip_mask(all_data)

			#plot flux
			f = [f[1]for f in all_data] #fluxes
			f = np.nan_to_num(f)
			m = np.mean(f) #get median to normalize
			print m
			plt.plot(times,f / m + offset,label = str(r))
			#calculate appropriate shift for next lightcurve
			offset += 2*(max(f)/m - 1)
			print max(f),m,offset

			for k in range(len(all_data)):
				output.write(','.join([str(z) for z in all_data[k]] + [str(cent_mask[k]),str(clip_mask[k])]) + '\n')
	plt.legend(loc = 0)
	plt.savefig('fits_analysis_data/lightcurves_' + centroid_method + '_' + bckg_method + '.png')

get_lightcurve('col','corners')
get_lightcurve('2dgauss','corners')

#light_curve(aor,frames,times,backgrounds,centroids,r=3) #put all together and get lightcurve



