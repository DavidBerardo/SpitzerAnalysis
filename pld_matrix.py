from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import batman
import pandas as pd
from scipy import interpolate 
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import sys
import time
import progressbar
from scipy.signal import resample

global transit_model
global N
global data
global times
global fname

#compute eclips/transit model
#initiate a batman model for each planet
def generate_transit_function(times,per,t0,rp,a,inc,ecc,w,limb_dark = 'linear',u = [0.14],phase_secondary=0.5,eclipse=False):
	params = batman.TransitParams()
	params.per = per
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = ecc
	params.w = w
	params.limb_dark = limb_dark
	params.u = u
	params.t0 = t0
	params.fp = 1
	params.t_secondary = t0 + per*0.5*(1 + 4/np.pi*ecc*np.cos(w*np.pi/180))

	#TODO change this so it puts a lot of points near the transit/eclipse
	t = np.linspace(times.iloc[0] - 0.5,times.iloc[-1] + 0.5,10000)

	if eclipse:
		#setup model to appropriate type, eclipse or primary
		model = batman.TransitModel(params,t,transittype='secondary')
		#generate lightcurve
		#with fp = 1, out-of-eclispe has a level of 2, so subtract this off to get eclipse variation (deltaFlux)
		lightcurve = model.light_curve(params) - 2
		#convert lightcurve to function
		lightcurve = interpolate.interp1d((t - times.iloc[0])*24,lightcurve)
		return lightcurve

	else:
		#setup model to appropriate type, eclipse or primary
		model = batman.TransitModel(params,t)
		#subtract out of transit level to variation due to transit (deltaFlux)
		lightcurve = (model.light_curve(params) - 1)
		lightcurve = lightcurve/abs(np.min(lightcurve))
		#convert lightcurve to function
		lightcurve = interpolate.interp1d((t - times.iloc[0])*24,lightcurve)
		return lightcurve

def find_solution(phase_offset,pts_per_bin = 1,show = False):
	if pts_per_bin == 1:
		binned_data = data
	else:
		#bin data points
		binned_data = pd.DataFrame()
		n = pts_per_bin
		for i in range(len(list(data))):
			#binned_data[i] = np.array([np.mean(data[i][n*j:min(n*(j+1),len(data[0]))]) for j in range(int(len(data)/n))])
			binned_data[i] =  np.convolve(data[i],np.ones((n,))/n,mode='valid')[::n]
		#print 'Binning done'
	times = binned_data[0]
	times = (times - times.iloc[0])*24
	flux = binned_data[1]
	flux = flux/np.mean(flux)

	#transit data
	transit_data = transit_model(times + phase_offset)

	#create matrix of data
	#x = np.concatenate((np.asarray(binned_data.iloc[:,2:2+N]),np.asarray([transit_data]).T, np.ones((len(times),1)),np.asarray([times]).T, np.asarray([times**2]).T),axis = 1)
	x = np.concatenate((np.asarray(binned_data.iloc[:,2:2+N]),np.asarray([transit_data]).T,np.asarray([times]).T, np.asarray([times**2]).T),axis = 1)

	#get coefficient matrix
	#solves x*a = data
	solution = np.dot(np.linalg.pinv(x),flux)
	result = np.dot(x,solution)
	residuals = flux - result
	error = np.std(residuals)

	if show:
		depth = solution[-3]
		print solution
		#plot data and model
		fig = plt.figure()
		plt.scatter(times,flux,s = 3)
		plt.scatter(times,result,s = 3)
		plt.savefig('binned_data_with_model.pdf')
		
		#plot data - model + transit
		fig = plt.figure()
		plt.scatter(times,1 + residuals + depth * transit_data,s = 2)
		plt.plot(times,1 + depth * transit_data, c = 'C1')
		plt.savefig('transit_signal.pdf')

		#plot pixel coefficients
		fig = plt.figure()
		coeffs = np.reshape(solution[:-3],(int(np.sqrt(N)),int(np.sqrt(N))))
		plt.imshow(coeffs)
		plt.colorbar()
		plt.show()

	#return chi_sq as well as value of constants
	return [chisquare(result,flux)[0],solution]

#plt.scatter(times,data[1]/np.mean(data[1]) - 1)
#plt.plot(times,transit_model(times),color = 'C1')
#plt.show()

def find_best_offset():
	print "Finding best offset"
	offset = np.linspace(-3,3,200)
	best = 100000
	best_offset = 1e10

	#find best offset
	pbar = progressbar.ProgressBar()
	for i in pbar(offset):
		chisq = find_solution(i)[0]
		#plt.scatter(i,chisq,color = 'C1')
		if chisq < best:
			best, best_offset = chisq, i

	#find_solution(best_offset,show=True)
	print 'Best offset: ' + str(best_offset)
	return best_offset

#best_offset = find_best_offset()
#find_solution(best_offset,show=True)

#bin the data, calculate pld coeffs
#apply coeffs to unbinned data
#then look at scatter as a function of binned residuals
def find_best_bin_size(best_offset,record_results = False):
	print "Finding best bin size"
	pts_per_bin = [1] + [2 + 4*i for i in range(65)]
	flux = data[1]
	best_bin_size_chi_sq, best_chi_sq = -1, 1e10
	best_bin_size_res_std, best_res_std = -1, 1e10
	pbar = progressbar.ProgressBar()
	residual_stds = [] #keep track of std of (data - model) on unbinned residuals
	white_chi_sqs = [] #keep track of sigma vs bins
	best_sigma_res = [] #keep track of best sigma vs binned residuals

	if record_results:
		best_bin_results = open('best_bin_results/' + fname.split('/')[-1],'w')

	for i in pbar(pts_per_bin):
		solution = find_solution(best_offset,pts_per_bin = i)[1]
		
		times = data[0]
		times = (times - times.iloc[0])*24
		flux = data[1]
		flux = flux/np.mean(flux)
		transit_data = transit_model(times + best_offset)
		
		#x = np.concatenate((np.asarray(data.iloc[:,2:2+N]),np.asarray([transit_data]).T, np.ones((len(times),1)), np.asarray([times]).T, np.asarray([times**2]).T),axis = 1)
		x = np.concatenate((np.asarray(data.iloc[:,2:2+N]),np.asarray([transit_data]).T, np.asarray([times]).T, np.asarray([times**2]).T),axis = 1)

		#applying ci coefficients to unbinned data
		result = np.dot(x,solution)
		residuals = flux - result

		residual_stds.append(np.std(residuals))

		#now bin residuals and calcuate std for each
		residual_pts_per_bin = 1
		bin_vals = [1]
		stds = [np.std(residuals)]

		#this is currently the slowest part of all this
		while True:
			#residuals = [np.mean(residuals[2*j:2*(j+1)]) for j in range(int(len(residuals)/2))]
			residuals = np.convolve(residuals,np.ones((2,))/2,mode='valid')[::2]
			if len(residuals) <= 16: 
				break
			residual_pts_per_bin *= 2
			bin_vals.append(residual_pts_per_bin)
			stds.append(np.std(residuals))

		#write out header for results file
		if i == 1 and record_results:
			best_bin_results.write('pts_per_bin,res_std,white_noise_ch_sq,' + ','.join(['res_bin_' + str(zz) + '_std' for zz in bin_vals]) + '\n')

		#compare a line of slope -.5 to standard deviations and get chi square		
		model = [stds[z] * bin_vals[z]**(-0.5) for z in range(len(stds))]
		chi_sq = chisquare(model,stds)[0]
		white_chi_sqs.append(chi_sq)

		if record_results:
			best_bin_results.write(str(i) + ',' + str(residual_stds[-1]) + ',' + str(chi_sq) + ',' + ','.join([str(zz) for zz in stds]) + '\n')
		
		if chi_sq < best_chi_sq:
			best_chi_sq, best_bin_size_chi_sq = chi_sq, i 
			best_sigma_res = stds

		if residual_stds[-1] < best_res_std:
			best_res_std, best_bin_size_res_std = residual_stds[-1], i


	if record_results:
		fig = plt.figure(figsize = (10,8))
		#plt.suptitle(fname.split('/')[-1][:-5])

		ax = fig.add_subplot(2,2,1)
		plt.scatter(pts_per_bin,residual_stds,s=2,label = '')
		plt.scatter(best_bin_size_res_std,residual_stds[pts_per_bin.index(best_bin_size_res_std)],s=10,color = 'C1',label = 'best res std')
		plt.scatter(best_bin_size_chi_sq,residual_stds[pts_per_bin.index(best_bin_size_chi_sq)],s=10,color='C3',label = 'best chi sq')
		plt.legend(loc = 0, frameon = False)
		plt.xlabel('pts per bin')
		plt.ylabel(r'$\mathrm{Residual}\ \sigma$')
		span = max(residual_stds) - min(residual_stds)
		plt.ylim([min(residual_stds) - 0.1 * span,max(residual_stds) + 0.1 * span])

		ax = fig.add_subplot(2,2,2)
		plt.scatter(pts_per_bin,white_chi_sqs,s=2,label = '')
		plt.scatter(best_bin_size_res_std,white_chi_sqs[pts_per_bin.index(best_bin_size_res_std)],s=10,color='C1',label= 'best res std')
		plt.scatter(best_bin_size_chi_sq,white_chi_sqs[pts_per_bin.index(best_bin_size_chi_sq)],s=10,color='C3',label = 'best chi sq')
		plt.legend(loc = 0, frameon = False)
		plt.xlabel('pts per bin')
		plt.ylabel(r'$\mathrm{White\ Noise}\ \chi^2$')
		span = max(white_chi_sqs) - min(white_chi_sqs)
		plt.ylim([min(white_chi_sqs) - 0.1 * span,max(white_chi_sqs) + 0.1 * span])

		ax = fig.add_subplot(2,2,3)
		plt.scatter(residual_stds,white_chi_sqs,s=2,label='')
		plt.scatter(residual_stds[pts_per_bin.index(best_bin_size_res_std)],white_chi_sqs[pts_per_bin.index(best_bin_size_res_std)],s=10,color = 'C1',label = 'best res std')
		plt.scatter(residual_stds[pts_per_bin.index(best_bin_size_chi_sq)],white_chi_sqs[pts_per_bin.index(best_bin_size_chi_sq)],s=10,color='C3',label = 'best chi sq')
		plt.xlabel(r'$\mathrm{Residual}\ \sigma$')
		plt.ylabel(r'$\mathrm{White\ Noise}\ \chi^2$')
		spanx = max(residual_stds) - min(residual_stds)
		spany = max(white_chi_sqs) - min(white_chi_sqs)
		plt.xlim([min(residual_stds) - 0.1 * spanx,max(residual_stds) + 0.1 * spanx])
		plt.ylim([min(white_chi_sqs) - 0.1 * spany,max(white_chi_sqs) + 0.1 * spany])
		plt.legend(loc = 0, frameon = False)

		ax = fig.add_subplot(2,2,4)
		x = np.linspace(np.log10(bin_vals)[0],np.log10(bin_vals)[-1],100)
		y = np.log10(best_sigma_res)[0] - 0.5 * x
		plt.plot(x,y,c = 'C1')
		plt.scatter(np.log10(bin_vals),np.log10(best_sigma_res),s=10)
		plt.xlabel(r'$\mathrm{log_{10}(residual pts per bin)}$')
		plt.ylabel(r'$Log_{10}\sigma$')



		plt.tight_layout()
		plt.savefig('best_bin_results/' + fname.split('/')[-1][:-5] + '.pdf')



	print 'Best number of pts per bin by res std: ' + str(best_bin_size_res_std)
	print 'Best number of pts per bin by chi sq: ' + str(best_bin_size_chi_sq)

	return best_bin_size_chi_sq, best_chi_sq, best_bin_size_res_std, best_res_std

if __name__ == '__main__':
	#get data
	aor = 62714368
	global data
	r = '3.0'
	centroid = '2dgauss'
	n = 3
	fname = 'lightcurves/' + str(n) + 'x' + str(n) +  '/r' + str(aor) + '_photometry_radius_' + r + '_' + centroid + '.data'
	data = pd.read_csv(fname,header=None)

	times = data[0]

	#number of pixels used in pld
	N = 9

	transit_model = generate_transit_function(times,per = 21.0580, t0 = 2457611.1328 - 2400000.5 , rp = 0.03207, a=25.69,inc=88.48,ecc=0.18,w=53.0,eclipse = False)
	times = (times - times.iloc[0])*24

	best_offset = find_best_offset()
	best_bin_size_chi_sq, best_chi_sq, best_bin_size_res_std, best_res_std =  find_best_bin_size(best_offset)
	find_solution(best_offset,best_bin_size_chi_sq,show=True)
	#find_solution(-0.10552763819095468,42,show=True)