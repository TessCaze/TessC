import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import statistics
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from astropy.io import fits
import glob
#from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit



#fits.info(image_file)
#print(type(image_data))
#print(fits.info(image_file))
#print(image_data.shape) #Shape of ciber files are is 328 by 328
'''
Step 1: take a look at the images a a whole
Step 2: take the array of images to put through code
Step 3: run each image through Gaussian Guassian_filter
Step 4: find brightest pixel in each images
Step 5: put a red line where the x data points
Step 6: cut the array out of images
Step 7: plot the data
Step 8: fit a gaussian
Step 9: print results of data parameters

Step 10: make an array of the results
Step 11: Plot the results
Step 10:Find the best fit curve image.
'''

'''
Here I am testing code with ONE image fitting a Gaussian function
'''

files = glob.glob('/data/focus_sims/ciber_data/fits_files')
image_hdus = []

# for f in range(5):
og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_14.FITS') #%.2d.FITS' % int(f+11))
image_hdus.append(og_im[0].data)
image_data_filt = gaussian_filter(og_im[0].data, 5)
result = np.where(image_data_filt == np.amax(image_data_filt)) #returns indices
x_data = image_data_filt[:,result[0]] #result[0] is the single y data point
ylist = [item for sublist in x_data for item in sublist] #flattens lists to one list
x_array = np.arange(len(x_data)) #numbers from 0 to 327



def gaussian_func(x, amp , mean, std):
    return amp*np.exp(-(x-mean)**2/(2*std**2))

popt, pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])


plt.figure()
plt.plot(x_array, ylist, 'go', markersize = 4, label = 'Image data')
plt.plot(x_array, gaussian_func(x_array, popt[0], popt[1], popt[2]), 'b', label='Best fit')
plt.legend()
# statistics.stdev(sample)

plt.savefig('/home/time_user/TessC/fits_plots/testing_stamp_14.png') #%.2d.png' % int(f+11))

print("by curve_fit:")
print("amplitude = ", round(popt[0],2))
print("mean = ", round(popt[1],2))
print("std = ", round(popt[2],2))

print("by hand:")
print("amp = ", round(max(ylist),2))
print("mean = ", round(np.mean(ylist),2))
print("std = ", round(np.std(ylist),2))


'''
Here I am testing code with blob images fitting a Gaussian function in each
'''
# files = glob.glob('/data/focus_sims/ciber_data/fits_files')
# image_hdus = []
#
# def gaussian_func(x, amp , mean, std):
#     return amp*np.exp(-(x-mean)**2/(2*std**2))
#
# for f in range(5):
#     og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_%.2d.FITS' % int(f+11))
#     image_hdus.append(og_im[0].data)
#     image_data_filt = gaussian_filter(og_im[0].data, 5)
#     result = np.where(image_data_filt == np.amax(image_data_filt)) #returns indices
#     x_data = image_data_filt[:,result[0]] #result[0] is the single y data point
#     ylist = [item for sublist in x_data for item in sublist] #flattens lists to one list
#     x_array = np.arange(len(x_data)) #numbers from 0 to 327
#
#     popt, pcov = curve_fit(gaussian_func, x_array, ylist)
#
#     plt.figure()
#     plt.plot(x_array, ylist, 'go', markersize = 4, label = 'Image data')
#     plt.plot(x_array, gaussian_func(x_array, popt[0], popt[1], popt[2]), 'b', label='Best fit')
#     plt.legend()
#     # statistics.stdev(sample)
#     print("For subgrid_stamp_%.2d.png the Statistics are:" % int(f+11))
#     print("amplitude = ", round(popt[0],2))
#     print("mean = ", round(popt[1],2))
#     print("std = ", round(popt[2],2))
#
#     plt.savefig('/home/time_user/TessC/fits_plots/Gauss_plot_stamp_%.2d.png' % int(f+11))
