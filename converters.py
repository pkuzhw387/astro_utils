import numpy as np

def flux2mag(flux, f_err):
	# Function to convert flux to magnitudes. 
	# Parameters:
	# -------------
	# 
	# flux: array
	# 		flux array to be converted
	# 
	# f_err: array
	# 		flux error array to be converted
	# 		
	# 
	# Returns:
	# ------------
	# 
	# mag: array
	# 		magnitude array
	# 		
	# mag_err: array
	# 		magnitude error array
	# 		
	# 
	f0 = np.max(flux)
	err_0 = f_err[np.argmax(flux)]
	mag = -2.5 * np.log10(flux / f0)
	mag_err = 2.5 / np.log(10) * ((err_0 / f0)**2 + (f_err / flux)**2)**0.5
	return mag, mag_err