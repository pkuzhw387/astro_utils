import numpy as np
from matplotlib import pyplot as plt


def plot_with_shaded_sigma(x, y, sigma, color=None, label=None, xlabel=None, ylabel=None):
	# plot the scaling relation between the data x and y with the shaded area
	# denoting 1 \sigma scatter.
	plt.plot(x, y, color=color, linewidth=5.0, label=label)
	plt.plot(x, y - sigma, color=color, linewidth=1.0)
	plt.plot(x, y + sigma, color=color, linewidth=1.0)
	plt.fill_between(x, y - sigma, y + sigma, color=color, alpha=0.25, label=r'1$\sigma$ scatter')
	plt.legend(loc='upper right')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	


def scaling_relation_plot(x, y, bins=15, color='blue', refer_val='median', label=None, xlabel=None, ylabel=None, small_bin_num=10):
	x_bins = np.linspace(np.min(x), np.max(x), bins)
	x_binned = []
	y_binned = []
	y_sigma = []
	if refer_val == 'median':
		for i in range(len(x_bins) - 1):
			ind_binned = np.intersect1d(np.where(x >= x_bins[i]), np.where(x < x_bins[i + 1]))
			if len(ind_binned) == 0 or len(ind_binned) == 1:
				print x_bins[i], x_bins[i+1]
			x_binned.append(np.mean(x[ind_binned]))
			y_binned.append(np.median(y[ind_binned]))
			y_sigma.append(np.std(y[ind_binned]))
			if len(ind_binned) < small_bin_num:
				plt.scatter(x[ind_binned], y[ind_binned])
	elif refer_val == 'mean':
		for i in range(len(x_bins) - 1):
			ind_binned = np.intersect1d(np.where(x >= x_bins[i]), np.where(x < x_bins[i + 1]))
			x_binned.append(np.mean(x[ind_binned]))
			y_binned.append(np.mean(y[ind_binned]))
			y_sigma.append(np.std(y[ind_binned]))
			if len(ind_binned) < small_bin_num:
				plt.scatter(x[ind_binned], y[ind_binned])

	x_binned = np.array(x_binned)
	y_binned = np.array(y_binned)
	y_sigma = np.array(y_sigma)

	plot_with_shaded_sigma(x_binned, y_binned, y_sigma, color=color, label=label, xlabel=xlabel, ylabel=ylabel)
