from javelin_utils import javelin_run
import numpy as np


Objects = ['arp151', 'ic4218', 'mcg6', 'mrk142', 'mrk202', 'mrk290',\
		  'mrk766', 'mrk871', 'mrk1310', 'ngc4748', 'ngc5548', 'ngc6814',\
		  'sbs']

z = np.array([0.059, 0.01933, 0.00775, 0.04494, 0.02102, 0.02958,\
	 0.01293, 0.03366, 0.01941, 0.01463, 0.01718, 0.00521,\
	 0.02787])

tau0 = np.array([3.45, 10.0, 10.0, 2.76, 3.05, 10.0,\
		6.78, 10.0, 3.6, 6.3, 4.17, 6.46,\
		2.18])

test = javelin_run(obj_list=Objects, model='pow-law')

test.lag_stat(param_dir='/Users//zhanghaowen/Desktop/AGN/BroadBand_RM/data/lamp2008reduced/',\
			  model='pow-law', redshift=z, tau0=tau0, cont_band='r', line_band=['g'])