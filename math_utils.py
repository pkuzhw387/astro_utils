# This is a mathematical and astronomical utility package made by Haowen Zhang for Python.

import numpy as np 
import scipy

def convolve(data, kernel, scale=1.0, center=False, edge_truncate=True):
	n = len(data)
	k = len(kernel)

	result = np.zeros(n)
	if edge_truncate == False:
		if center == True:
			for t in range(n):
				if k / 2 <= t <= n - 1 - k / 2:
					result[t] = np.sum(data[t - k / 2:t + k - k / 2] * kernel[0:k])
		else:
			for t in range(n):
				if t >= k - 1:
					result[t] = np.sum(data[t - k + 1:t + 1] * kernel[0:k])


	else:
		if center == True:
			for t in range(n):
				for i in range(k):
					if t + i - k / 2 < 0:
						result[t] += data[0] * kernel[i]
					elif t + i - k / 2 >= n:
						result[t] += data[n - 1] * kernel[i]
					else: 
						result[t] += data[t + i - k / 2] * kernel[i]
		
		else:
			for t in range(n):
				for i in range(k):
					if t - i < 0:
						result[t] += data[0] * kernel[i]
					else: 
						result[t] += data[t - i] * kernel[i]

	result = result / scale
	return result





