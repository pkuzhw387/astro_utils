# This is a mathematical and astronomical utility package made by Haowen Zhang for Python.
from math_utils import convolve
import numpy as np 
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import quad
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt 
import os

# h = 6.63e-27
# c = 3.0e10
# kb = 1.3806505e-16


c = 3.0e8

def gauss(x, params):
	A = params[0]
	x0 = params[1]
	sigma = params[2]
	return A / ((2*np.pi)**0.5 * sigma) * np.exp(-(x - x0)**2 / (2 * sigma**2))

def line_err2(x, params, params_err, flux=None):
	A = params[0]
	x0 = params[1]
	sigma = params[2]

	A_err = params_err[0]
	x0_err = params_err[1]
	sigma_err = params_err[2]

	if flux is None:
		flux = gauss(x, params)
	var = (flux * A_err / A)**2
	# print A
	var += (flux * (x - x0) * x0_err / sigma**2)**2
	var += (flux * ((x - x0)**2 / sigma**3 - 1 / sigma) * sigma_err)**2
	return var

def power_law(x, f0, alpha):
	return f0 * (x / 3000.0)**alpha

def flux_integrand(x, flux, response):
	return flux(x) * response(x) * x
	#return flux(x) * response(x)

def var_integrand(x, var, response):
	return var(x) * response(x) * x**3
	#return ivar(x) * response(x)

def norm_integrand(x, response):
	return response(x) / x
	#return response(x)

AB_filter_names = ['u', 'g', 'r', 'i', 'z']
Vega_filter_names = ['U', 'B', 'V', 'R', 'I']



class Spectra(object):
	def __init__(self, spec_dir, spec_format='txt', spec_source=None, frame='observer'):
		# Spectra object.
		# Parameters:
		# -------------------
		# spec_dir: string
		# 		Input spectra directory.
		# 
		# spec_format: string
		# 		Input spectra format. optional from ['txt', 'fits', 'dat']
		# 
		# spec_source: string, optional
		# 		The source from which the spectra are obtained. default: None
		# 		
		# frame: string, optional
		# 		specify the frame in which the wavelengths are measured.
		# 		'restframe' means the wavelengths are in restframe, i.e. divided by (1 + z);
		# 		'observer' means the wavelengths are not corrected by (1 + z) factor. 
		# 		default: 'observer'
		self.spec_dir = spec_dir
		self.spec_format = spec_format
		self.spec_source = spec_source
		self.frame = frame
		self.unit = {'wav': 'Angstrom'}

	def read_spec(self):
		# currently only the 'LAMP' mode is implemented, I will add other modes (for different spectra sources) later on.
		if self.spec_source == 'LAMP':
			self.no_air_mass = ['ic4218', 'mcg6', 'mrk290', 'mrk871']
			self.unit['flux'] = '10^-15 erg/s/cm^2/A'
			self.unit['err'] = '10^-15 erg/s/cm^2/A'
			self.spec_format = 'txt'
			self.fluxes = []
			self.wavs = []
			self.errs = []
			spec_list = os.listdir(self.spec_dir)
			for i in range(len(spec_list)):
				if os.path.splitext(spec_list[i])[1] == '.dat':
					self.obj_name = spec_list[i].split('.')[0]
					if self.obj_name in self.no_air_mass:
						self.info_df = pd.read_csv(os.path.join(self.spec_dir, spec_list[i]), header=None, \
											  names=['filename', 'JD'], skiprows=1, delim_whitespace=True, \
											  usecols=[0, 1], dtype={'filename': str, 'JD': np.float64})
					else:
						self.info_df = pd.read_csv(os.path.join(self.spec_dir, spec_list[i]), header=None, \
											  names=['filename', 'JD'], skiprows=1, delim_whitespace=True, \
											  usecols=[0, 3], dtype={'filename': str, 'JD': np.float64})
					self.JD = np.array(self.info_df['JD'])
					self.nepoch = len(self.JD)
			for ind, spec in enumerate(self.info_df['filename']):
				spec_path = os.path.join(self.spec_dir, spec)
				spec_df = pd.read_csv(spec_path, header=None, names=['wav', 'flux', 'err'], usecols=[0, 1, 2], \
									  delim_whitespace=True, dtype=np.float64)
				self.wavs.append(np.array(spec_df['wav']))
				self.fluxes.append(np.array(spec_df['flux']))
				self.errs.append(np.array(spec_df['err']))

			self.wavs = np.array(self.wavs)
			self.fluxes = np.array(self.fluxes)
			self.errs = np.array(self.errs)

			# print self.wavs
			# print self.fluxes
			# print self.errs
		
		elif self.spec_source == 'IDL':
			# If the spectra come from IDL, then it is already fitted.
			self.spec_format = 'fits'
			self.spec_list = []
			self.wavs = []
			self.fluxes = []
			self.line_fluxes = []
			self.errs = []
			self.line_errs = []

			raw_spec_list = os.listdir(self.spec_dir)
			for ind in range(len(raw_spec_list)):
				if raw_spec_list[ind] == '.DS_Store' or os.path.splitext(raw_spec_list[ind])[1] != '.fits':
					continue
				else:
					self.obj_name = os.path.splitext(raw_spec_list[ind])[0]
					self.spec_list.append(raw_spec_list[ind])
					spec_data = fits.open(os.path.join(self.spec_dir, raw_spec_list[ind]))

					wave = spec_data[1].data['WAVE'][0]
					self.wavs.append(wave)
					self.fluxes.append(spec_data[1].data['FLUX'][0])
					self.errs.append(spec_data[1].data['ERR'][0])

					self.f0, self.alpha = spec_data[1].data['CONTI_FIT'][0][0:2]
					params = spec_data[1].data['LINE_FIT'][0]
					params_err = spec_data[1].data['LINE_FIT_ERR'][0]

					tot_line_flux = np.zeros(len(wave))
					tot_line_err = np.zeros(len(wave))

					Hb_sigma = [params[44 + 3*i] for i in range(3)]
					MgII_sigma = [params[26 + 3*i] for i in range(3)]

					for i in range(3):
						line_flux = gauss(np.log(wave), params[24 + 3 * i : 27 + 3 * i])
						line_flux = gauss(np.log(wave), params[24 + 3 * i : 27 + 3 * i])
						line_var = line_err2(np.log(wave), params[24 + 3 * i : 27 + 3 * i], params_err[24 + 3 * i : 27 + 3 * i], line_flux)
						tot_line_flux += line_flux
						tot_line_err += line_var
					tot_line_err = tot_line_err**0.5

					self.line_fluxes.append(tot_line_flux)
					self.line_errs.append(tot_line_err)
			self.nepoch = len(self.spec_list)
			self.wavs = np.array(self.wavs)
			self.fluxes = np.array(self.fluxes)
			self.errs = np.array(self.fluxes)
			self.line_fluxes = np.array(self.line_fluxes)
			self.line_errs = np.array(self.line_errs)

	def mean_calc(self):
		# Function to calculate rms and mean spectra. 
		
		# rms calculation
		self.rms = {}
		print "waves: ", self.wavs
		self.rms['wav'] = self.wavs[0, :]
		self.rms['flux'] = np.mean(self.fluxes**2, axis=0)**0.5
		self.rms['err'] = np.sum(abs(self.fluxes) * self.errs, axis=0) / self.rms['flux'] / self.fluxes.shape[0]
		
		# mean calculation
		self.mean = {}
		self.mean['wav'] =  self.wavs[0, :]
		self.mean['flux'] = np.mean(self.fluxes, axis=0)
		self.mean['err'] = np.mean(self.errs**2, axis=0)**0.5

	def mean_record(self, out_dir=None, plot=True):
		# Function to write rms and mean spectra to .txt files
		# Parameters:
		# ------------
		# out_dir: str, optional
		# 		specify the output directory.
		if out_dir is None:
			out_dir = self.spec_dir
		out_path = os.path.join(out_dir, (self.obj_name + '_rms.txt'))
		out_data = np.transpose(np.array([self.rms['wav'], self.rms['flux'], self.rms['err']]))
		out_df = pd.DataFrame(out_data, columns=['wav', 'flux', 'err'])
		out_df.to_csv(out_path, columns=['wav', 'flux', 'err'], header=None, index=False, sep=' ')

		out_path = os.path.join(out_dir, (self.obj_name + '_spec_rms.png'))
		if plot:
			plt.plot(self.rms['wav'], self.rms['flux'])
			plt.savefig(out_path)
			plt.close()			

	def light_curve_calc(self, filter_sys='AB', filter_set='SDSS', extra_wav_range=None, plot=True, mode='total'):
		# Function to calculate the light curves under a given system and set of filters. 
		# Parameters:
		# ------------
		# filter_sys: str, optional
		# 		The filter system to be used. 
		# 		'AB': ugriz system
		# 		'Vega': UBVRI system
		# 		default: 'AB'
		# 		
		# filter_set: str, optional
		# 		The specific filter set to be used.
		# 		e.g. 'SDSS' or 'CFHT'
		# 		default: 'SDSS'
		# 		
		# extra_wav_range: dict, optional
		# 		extra wavelength range limits on different bands.
		# 		e.g. {'g': [5500, 5600], 'i': [7500, 7600]}
		# 		default: None
		# 		
		# plot: bool, optional
		# 		Whether to make the plot of the light curves.
		# 		Default: True
		# 		
		# mode: str, optional
		# 		'total': calculate total flux (line + continuum)
		# 		'cont': only continuum
		# 		'line': only line
		

		if mode == 'total':
			# spectra interpolation
			self.interp_fluxes = [interp1d(self.wavs[i], self.fluxes[i], kind='nearest') for i in range(self.nepoch)]
			self.interp_errs = [interp1d(self.wavs[i], self.errs[i], kind='nearest') for i in range(self.nepoch)]
			self.interp_vars = [interp1d(self.wavs[i], self.errs[i]**2, kind='nearest') for i in range(self.nepoch)]
		elif mode == 'line':
			self.interp_fluxes = [interp1d(self.wavs[i], self.line_fluxes[i], kind='nearest') for i in range(self.nepoch)]
			self.interp_errs = [interp1d(self.wavs[i], self.line_errs[i], kind='nearest') for i in range(self.nepoch)]
		elif mode == 'cont':
			self.interp_fluxes = [interp1d(self.wavs[i], self.fluxes[i] - self.line_fluxes[i], kind='nearest') for i in range(self.nepoch)]




		# filter curve interpolation and record the filter wavelength ranges
		filter_range = {}
		filter_curve = {}

		if filter_sys == 'AB':
			if filter_set is None or filter_set == 'SDSS':
				nskip_rows = 6
				filter_dir = '/Users/zhanghaowen/Desktop/AGN/BroadBand_RM/filters/SDSS/'
			elif filter_set == 'CFHT':
				nskip_rows = nskip_rows = 0
				filter_dir = '/Users/zhanghaowen/Desktop/AGN/BroadBand_RM/filters/CFHT/'

			for i in range(len(AB_filter_names)):
				filter_path = filter_dir + AB_filter_names[i] + '.txt'
				filt_data = pd.read_csv(filter_path, header=None, names=['wavelength', 'response'], \
										usecols=[0, 1], delim_whitespace=True, dtype=np.float64, skiprows=nskip_rows)
				filter_curve[AB_filter_names[i]] = interp1d(filt_data['wavelength'], filt_data['response'], kind='nearest')
				if extra_wav_range is not None and AB_filter_names[i] in extra_wav_range.keys():
					filter_range[AB_filter_names[i]] = [np.max([filt_data['wavelength'][0], extra_wav_range[AB_filter_names[i]][0]]),\
														np.min([np.array(filt_data['wavelength'])[-1], extra_wav_range[AB_filter_names[i]][-1]])]
				else:
					filter_range[AB_filter_names[i]] = [filt_data['wavelength'][0], np.array(filt_data['wavelength'])[-1]]


		if not hasattr(self, 'lc'):
			self.lc = {}

		for filt in AB_filter_names:
			if not self.lc.has_key(filt):
				self.lc[filt] = {}

			self.lc[filt]['JD_%s' % mode] = []
			self.lc[filt]['flux_%s' % mode] = []
			self.lc[filt]['err_%s' % mode] = []

			for i in range(self.nepoch):
				# ignore the totally un-overlap filters.
				if filter_range[filt][0] >= self.wavs[i][-1] or filter_range[filt][-1] <= self.wavs[i][0]:
					self.lc[filt]['valid_tag_%s' % mode] = 0
					break
				else:
					# determine the wave length range used in interpolation. 
					wav_min = np.max([filter_range[filt][0], self.wavs[i][0]])
					wav_max = np.min([filter_range[filt][-1], self.wavs[i][-1]])
					tmp_flux = quad(flux_integrand, wav_min, wav_max, args=(self.interp_fluxes[i], filter_curve[filt]))[0] / \
							quad(norm_integrand, wav_min, wav_max, args=(filter_curve[filt]))[0] / c
					
					# for mode =='cont', no errs are calculated.
					if hasattr(self, 'interp_errs'):
						tmp_err = quad(flux_integrand, wav_min, wav_max, args=(self.interp_errs[i], filter_curve[filt]))[0] / \
								quad(norm_integrand, wav_min, wav_max, args=(filter_curve[filt]))[0] / c
					else:
						tmp_err = np.nan

					# for single epoch (e.g. rms or mean spectra), the object has no 'JD' attribute.
					if hasattr(self, 'JD'):
						self.lc[filt]['JD_%s' % mode].append(self.JD[i])
					else:
						self.lc[filt]['JD_%s' % mode].append(i)
					self.lc[filt]['flux_%s' % mode].append(tmp_flux)
					self.lc[filt]['err_%s' % mode].append(tmp_err)
					# if the filter range is totally enclosed in the range of the spectra, set the valid_tag as 1
					if filter_range[filt][0] >= self.wavs[i][0] or filter_range[filt][-1] <= self.wavs[i][-1]:
						self.lc[filt]['valid_tag_%s' % mode] = 1
					# if the filter range and the spectra wavelength have partial overlap, set the valid_tag as 0.5
					else:
						self.lc[filt]['valid_tag_%s' % mode] = 0.5


			self.lc[filt]['JD_%s' % mode] = np.array(self.lc[filt]['JD_%s' % mode])
			self.lc[filt]['flux_%s' % mode] = np.array(self.lc[filt]['flux_%s' % mode])
			self.lc[filt]['err_%s' % mode] = np.array(self.lc[filt]['err_%s' % mode])

	def light_curve_record(self, out_dir=None, plot=True):
		# Function to record the calculated light curves in txt format.
		# Parameters:
		# ------------
		# out_dir: str, optional
		# 		the output directory of the light curve. 
		# 		Default: self.spec_dir
		# plot: bool, optional
		# 		Whether to draw and save the light curve plots. 
		# 		Default: True
		if out_dir is None:
			out_dir = self.spec_dir

		for filt in AB_filter_names:
			if self.lc[filt]['valid_tag']:
				out_path = os.path.join(out_dir, (self.obj_name + '_' + filt + '_' + str(self.lc[filt]['valid_tag']) + '.txt'))
				out_data = np.transpose(np.array([self.lc[filt]['JD'], self.lc[filt]['flux'], self.lc[filt]['err']]))
				out_df = pd.DataFrame(data=out_data, columns=['JD', 'flux', 'err'])
				out_df.to_csv(out_path, columns=['JD', 'flux', 'err'], header=None, index=False, sep=' ')

			if plot: 
				plt.scatter(self.lc[filt]['JD'], self.lc[filt]['flux'], label=filt, s=5)

		out_path = os.path.join(out_dir, (self.obj_name + '_LightCurve.png'))
		if plot:
			plt.savefig(out_path)
			plt.close()

	def ratio_calc(self, tar_line='Hb', rec=True, cont_band='r', line_band='g'):
		# Function to calculate the line-to-continuum ratio of a given line. Only valid when the spec_source =='IDL'
		# Parameters:
		# ------------
		# tar_line: str, optional
		# 		The specific line to calculate ratios.
		# 		Default: 'Hb'
		# rec: bool, optional
		# 		Whether to record the ratio.
		# 		Default: True
		
		
		# Calculate fluxes of three modes
		self.light_curve_calc(plot=False, mode='total')
		self.light_curve_calc(plot=False, mode='line')
		self.light_curve_calc(plot=False, mode='cont')

		# abort if the light curve is not valid
		if self.lc[line_band]['valid_tag_line'] == 0 or\
		   self.lc[cont_band]['valid_tag_cont'] == 0:
		   print "invalid light curve input, unable to calculate the ratio."
		   return

		# calculate the line scale
		line_scale = self.lc[line_band]['flux_line'] / self.lc[cont_band]['flux_total']
		alpha = self.lc[line_band]['flux_cont'] / self.lc[cont_band]['flux_total']


		self.lc[line_band]['JD_ratio'] = self.lc[line_band]['JD_line']
		self.lc[line_band]['line_scale'] = line_scale
		self.lc[line_band]['alpha'] = alpha

		print "Object: ", self.obj_name
		print "line_scale is: ", line_scale
		print "alpha is: ", alpha







		

	






