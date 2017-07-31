import numpy as np
from javelin.zylc import get_data
# from javelin.lcmodel import Pmap_model
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from StructFunc import Structure_Function as SF
from numpy.linalg import LinAlgError
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import os


class javelin_run(object):
	# wrapper of JAVELIN Pmap_model, to facilitate Pmap running tests.
	# Parameters:
	# ------------
	# obj_list: list of str
	# 		The list of the objects for which the photo-RM tests are to run.
	# 
	# model: str, optional
	# 		Specific model of the stochastic process, 'DRW' or 'pow-law'.
	# 		Default: 'DRW'
	# 
	# transfunc: str, optional
	# 		Specific form of the tranfer function, 'top-hat', 'delta' or 'gaussian'.
	# 		Default: 'top-hat'
	# 		
	# dat_type: strm optional
	# 		Specify whether the flux or magnitude is used.
	# 		Default: 'flux'
	# 		
	# lc_dir: str, optional
	# 		The light curve storing path.
	# 		Default: None
	# 
	# spec_source: str, optional
	# 		Which project did the spectra and the light curves come from.
	# 		Default: 'LAMP'
	def __init__(self, obj_list, model='DRW', transfunc='top-hat', dat_type='flux',lc_dir=None, spec_source='LAMP'):
				 
		self.obj_list = obj_list
		self.model = model
		self.transfunc = transfunc
		self.dat_type = dat_type
		self.lc_dir = lc_dir
		self.spec_source = spec_source

	# Function to run the JAVELIN.Pmap_model.do_mcmc()
	# Parameters:
	# -------------
	# n_run: int, optional
	# 		Number of runs for each object.
	# 		Default: 200
	# 
	# lag_limit: bool, optional
	# 		Whether to use the spectroscopic lags to constrain the photo-RM lags.
	# 		Default: False
	# 		
	# fix_SF: bool, optional
	# 		Whether to fit and fix the DRW or PL SF parameters.
	# 		Default: False
	# 
	# fix_scales: bool, optional
	# 		Whether to fix the line scales and alphas during the mcmc.
	# 		Default: False
	# 		
	# redshift: array, optional
	# 		The redshift array of the objects.
	# 		Default: None
	# 		
	# tau0: array, optional
	# 		The spectroscopic lags array of the objects.
	# 		Default: None
	# 
	# line_scales: array, optional
	# 		The line scale array of the objects.
	# 		Default: None
	# 		
	# alpha: array, optional
	# 		The alpha (i.e. scale_hidden) array of the objects.
	# 		Default: None
	# 		
	# cont_band: str, optional
	# 		Name of the continuum band.
	# 		Default: 'r'
	# 
	# line_band: list of str, optional
	# 		Name of the line bands.
	# 		Default: ['g']
	# 		
	# valid_tag: int, optional
	# 		The tag used to pick out valid light curves from the directory. c.f. spec_utils.Spectra.light_curve.calc()
	# 		Default: 1
	# 		
	# multi_line: bool, optional
	# 		Whether to do multi-line photo-RM.
	# 		Default: False
	# 		
	# hascontlag: bool, optional
	# 		Whether to consider continuum lags.
	# 		Default: False
	# 		
	# baldwin: bool, optional
	# 		Whether to use Baldwin effect to calculate the line scales.
	# 		Default: False
	# 		
	# 		
	def run(self, n_run=200, lag_limit=False,fix_SF=False, fix_scales=False,redshift=None, tau0=None,\
			line_scales=None, alpha=None, cont_band='r', line_band=['g'], valid_tag=1, multi_line=False,\
			hascontlag=False, baldwin=False):
		if self.lc_dir is None:
			self.lc_dir = './'
		if redshift is not None:
			zp1 = np.array(redshift) + 1.0
		cont_path = [self.lc_dir + self.obj_list[i] + '_' + cont_band + '_' + str(valid_tag) for i in range(len(self.obj_list))]
		line_path = [[]] * len(line_band)

		for i in range(len(line_band)):
			line_path[i] = [self.lc_dir + self.obj_list[i] + '_' + line_band[i] + '_' + str(valid_tag) for i in range(len(self.obj_list))]

		self.cont_path = cont_path
		self.line_path = line_path

		for ind, obj in enumerate(self.obj_list):
			if multi_line:
				pass
			else:
				for lb in line_band:
					param_path = os.path.join(self.lc_dir, (obj + '_' + lb + '_' + model + '_params'))
					param_record = open(param_path, 'a+')
					n_tested = len(param_record.readlines())
					i = 0
					while True:
						if i + n_tested == 200:
							break
						print "This is the %d-th test of object %s.\n" % (i + 1 + n_tested, obj)
						chain_path = os.path.join(self.lc_dir, (obj + '_' + lb + '_' + model + '_chain'))
						logp_path = os.path.join(self.lc_dir, (obj + '_' + lb + '_' + model + '_logp'))
						if os.path.exists(chain_path):
							os.remove(chain_path)
						if os.path.exists(logp_path):
							os.remove(logp_path)	
						cy = get_data([cont_path, line_path], dat_type=self.dat_type)	
						cymod = Pmap_model(zydata=cy, GPmodel=model, hascontlag=hascontlag, baldwin=baldwin)
						try:
							if lag_limit:
								if zp1 is None or tau0 is None:
									print "limiting lags requires redshift and spectroscopic lags for the objects."
									return
								if fix_SF:
									cont_lc = pd.read_csv(cont_path, header=None, names=['MJD', dat_type, 'err'],\
														  usecols=[0, 1, 2], delim_whitespace=True, dtype=np.float64)
									SF_param = SF.SF_fit_params(np.array(cont_lc['MJD']), np.array(cont_lc[dat_type]),\
																np.array(cont_lc['err']), model=model, MCMC_step=2000)
									if fix_scales:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[0,0,1,1,1,1,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0,alpha[ind]])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[0,0,1,1,0,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,line_scales[ind],alpha[ind]])
									else:

										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[0,0,1,1,1,1,1], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0,0])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[0,0,1,1,0,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0])
								else:
									if fix_scales:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[1,1,1,1,1,1,0], \
													  	p_fix=[0,0,0,0,0,0,alpha[ind]])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind],\
													  	fixed=[1,1,1,1,0,0], \
													  	p_fix=[0,0,0,0,line_scales[ind],alpha[ind]])
									else:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path, tau0=tau0[ind], zp1=zp1[ind])

							else:
								if fix_SF:
									cont_lc = pd.read_csv(cont_path, header=None, names=['MJD', dat_type, 'err'],\
														  usecols=[0, 1, 2], delim_whitespace=True, dtype=np.float64)
									SF_param = SF.SF_fit_params(np.array(cont_lc['MJD']), np.array(cont_lc[dat_type]),\
																np.array(cont_lc['err']), model=model, MCMC_step=2000)
									if fix_scales:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[0,0,1,1,1,1,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0,alpha[ind]])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[0,0,1,1,0,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,line_scales[ind],alpha[ind]])
									else:

										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[0,0,1,1,1,1,1], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0,0])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[0,0,1,1,0,0], \
													  	p_fix=[SF_param['sigma_KBS09'],SF_param['tau'],0,0,0,0])
								else:
									if fix_scales:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[1,1,1,1,1,1,0], \
													  	p_fix=[0,0,0,0,0,0,alpha[ind]])
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path,\
													  	fixed=[1,1,1,1,0,0], \
													  	p_fix=[0,0,0,0,line_scales[ind],alpha[ind]])
									else:
										if baldwin:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path)
										else:
											cymod.do_mcmc(nburn=200, fchain=chain_path, flogp=logp_path)

						except LinAlgError:
							continue
						i += 1
						chain_file = open(chain_path, 'r')
						logp_file = open(logp_path, 'r')
						lag = []
						logp = []
						#get the lag values and corresponding logp from record files
						for line in chain_file.readlines():
							lag.append(atof(line.split()[2]))
						for line in logp_file.readlines():
							logp.append(atof(line.split()[0]))
						chain_file.close()
						logp_file.close()
						lag = np.array(lag)
						logp = np.array(logp)
						lag_range = []
						
						param_record.write(str(cymod.bfp[0]) + ',' + str(cymod.bfp[1]) + ',' + str(cymod.bfp[2]) + ',' + str(cymod.bfp[3]) + ',' + str(cymod.bfp[4]) + ',' + str(cymod.bfp[5]) + ',' + str(min(lag_range)) + ',' + str(max(lag_range)) + '\n')
						del cymod
						gc.collect()
						chain_file.close()
						logp_file.close()
					param_record.close()



	def lag_stat(self, param_dir='./', model=None, redshift=None, tau0=None, tau0_err=None, cont_band='r', line_band=['g'], color_coding=None):
		# Function to calculate restframe photo-RM lags from yielded params files.
		# Parameters:
		# -------------
		# param_dir: str, optional
		# 		The directory containing the param files.
		# 		Default: './'
		# 
		# model: str, optional
		# 		The model used to do photo-RM.
		# 		Default: None
		# redshift: array, optional
		# 		The redshift array of the objects.
		# 		Default: None
		# 		
		# tau0: array, optional
		# 		The spectroscopic lags array of the objects.
		# 		Default: None
		# 
		# tau0_err: array of shape (2, N), where N is the length of the obj_list. optional
		# 		The low and up error bars of each tau0
		# 		Default: None
		# 
		# cont_band: str, optional
		# 		Name of the continuum band.
		# 		Default: 'r'
		# 
		# line_band: list of str, optional
		# 		Name of the line bands.
		# 		Default: ['g']
		# 		
		# color_coding: array, optional
		# 		The property to be color mapped.
		# 		Default: None
		# 		
		# 		
		if not hasattr(self, 'cont_band'):
			self.cont_band = cont_band
		if not hasattr(self, 'line_band'):
			self.line_band = line_band

		self.lag_peak_set = [[]] * len(self.line_band)
		self.lag_cent_set = [[]] * len(self.line_band)
		self.err_set = [[]] * len(self.line_band)
		if redshift is None:
			redshift = np.ones(len(self.obj_list))
		if model is None:
			model = self.model
		zp1 = np.array(redshift) + 1.0


		for lb_ind in range(len(self.line_band)):
			record_path = os.path.join(param_dir, ('lag_estm_%s_band_%s' % (self.line_band[lb_ind], model)))
			record = open(record_path, 'w')
			record.write('#All the lags are in the restframe!\n')
			record.write('#obj_name lag_cent lag_peak lag_std err_up err_low\n')

			for ind in range(len(self.obj_list)):
				param_path = os.path.join(param_dir, (self.obj_list[ind] + '_' + model + '_params'))
				param = pd.read_csv(param_path, names=['lag', 'lag_min', 'lag_max'],\
									header=None, usecols=[2, 6, 7], sep=',',\
									dtype=np.float64)
				lag = np.array(param['lag'])
				lag_min = np.median(np.array(param['lag_min']))
				lag_max = np.median(np.array(param['lag_max']))
				bw_set = np.linspace(1,15,1000)
				lag_grid = np.linspace(0, 2.0 * lag_max, 20100)


				grid = GridSearchCV(KernelDensity(), {'bandwidth': bw_set}, cv=3)
				grid.fit(lag.reshape(-1,1))
				KDE = grid.best_estimator_
				dens_grid = KDE.score_samples(lag_grid.reshape(-1,1))
				dens_grid = np.exp(dens_grid)

				plt.plot(lag_grid, dens_grid)
				fig_path = os.path.join(param_dir, (self.obj_list[ind] + '_' + self.line_band[lb_ind] + '_' + model + 'KDE.png'))
				plt.savefig(fig_path)
				plt.close()

				lag_peak = lag_grid[np.argmax(dens_grid)]

				mask1 = dens_grid < 0.8 * np.max(dens_grid)
				mask2 = dens_grid < 0.5 * np.max(dens_grid)

				m1_lag = np.ma.array(lag_grid, mask=mask1)
				m2_lag = np.ma.array(lag_grid, mask=mask2)

				lag_cent = 0.5 * (np.min(m1_lag) + np.max(m1_lag))
				lag_std = (np.max(m2_lag) - np.min(m1_lag)) / (2 * (np.log(2))**0.5)

				print "object: %s, lag_peak=%f, lag_cent=%f\n" %(self.obj_list[ind], lag_peak, lag_cent)

				err_up = ((lag_max - lag_cent)**2 + lag_std**2)**0.5 / zp1[ind]
				err_low = ((lag_min - lag_cent)**2 + lag_std**2)**0.5 / zp1[ind]
				lag_cent = lag_cent / zp1[ind]
				lag_peak = lag_peak / zp1[ind]
				lag_std = lag_std / zp1[ind]

				record.write(self.obj_list[ind] + ' ' +\
							 str(lag_cent) + ' ' + str(lag_peak) + ' ' +\
							 str(lag_std) + ' ' + str(err_up) + ' ' + str(err_low) + '\n')


				self.lag_cent_set[lb_ind].append(lag_cent)
				self.lag_peak_set[lb_ind].append(lag_peak)
				self.err_set[lb_ind].append([err_low, err_up])

			record.close()
		self.lag_cent_set[lb_ind] = np.array(self.lag_cent_set[lb_ind])
		self.lag_peak_set[lb_ind] = np.array(self.lag_peak_set[lb_ind])
		self.err_set[lb_ind] = np.array(self.err_set[lb_ind]).reshape((2, len(self.obj_list)))
		iden_line = np.linspace(0, 1.5 * np.max(tau0), 5)
		plt.plot(iden_line, iden_line, color='black', label=r'$\tau_{JAVELIN}=\tau_{spec}$ line')

		plt.errorbar(x=tau0, y=self.lag_cent_set[lb_ind], xerr=tau0_err, yerr=self.err_set[lb_ind],\
					linewidth=0.0, elinewidth=1, marker='o', color='blue', label=('%s model cent lags vs. spec lags' % model))
		# plt.errorbar(x=tau0, y=self.lag_peak_set[lb_ind], xerr=tau0_err, yerr=self.err_set[lb_ind],\
		# 			linewidth=0.0, elinewidth=1, marker='o', color='red', label=('%s model peak lags vs. spec lags' % model))
		plt.legend()
		plt.xlabel(r'$\tau_{spec}$/day', fontsize=20)
		plt.ylabel(r'$\tau_{photo}$/day', fontsize=20)

		fig_path = os.path.join(param_dir, (self.line_band[lb_ind] + '_' + model + '_lag_plot.png'))
		plt.savefig(fig_path)
		plt.close()


		if color_coding is not None:
			plt.plot(iden_line, iden_line, color='black', label=r'$\tau_{JAVELIN}=\tau_{spec}$ line')

			plt.scatter(x=tau0, y=self.lag_cent_set[lb_ind], c=color_coding, cmap=cm.jet, label=('%s model cent lags vs. spec lags' % model))
			# plt.scatter(x=tau0, y=self.lag_peak_set[lb_ind], c=color_coding, cmap=cm.jet, label=('%s model peak lags vs. spec lags' % model))
			# plt.legend()
			plt.xlabel(r'$\tau_{spec}$/day', fontsize=20)
			plt.ylabel(r'$\tau_{photo}$/day', fontsize=20)
			cbar = plt.colorbar()
			cbar.set_label('line-to-continuum flux ratio', size=20)
			fig_path = os.path.join(param_dir, (self.line_band[lb_ind] + '_' + model + '_lag_plot_cmap.png'))
			plt.savefig(fig_path)
			plt.close()














