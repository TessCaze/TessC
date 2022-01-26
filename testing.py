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
Step 11: cut vertical array out of images
Step 12: plot the data
Step 13: fit a gaussian

Step 14: print FWHM
Step 15: Plot the results
Step 16:Find the best fit curve image.
'''

'''
Here I am checking where the data is being cut from the image (Vertical cut & Horizontal cut)
'''
#
# files = glob.glob('/data/focus_sims/ciber_data/fits_files')
# image_hdus = []
#
# def gaussian_func(x, amp , mean, std):
#     return amp*np.exp(-(x-mean)**2/(2*std**2))
#
# #for f in range(5):
# og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_11.FITS')
# image_hdus.append(og_im[0].data)
# image_data_filt = gaussian_filter(og_im[0].data, 5)
# result = np.where(image_data_filt == np.amax(image_data_filt))
# #print(result) #x and y coords of max
# x_data = image_data_filt[:,result[1]]
# xlist = [item for sublist in x_data for item in sublist] #flattens list into one list
#
# y_data = image_data_filt[result[0],:]
# ylist = [item for sublist in x_data for item in sublist]
# x_array = np.arange(len(x_data)) #numbers from 0 to 327 for x-axis of guassian plot
#
# h_popt, h_pcov = curve_fit(gaussian_func, x_array, xlist, p0 = [max(xlist), np.mean(xlist), np.std(xlist)])
# v_popt, v_pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('H and V data cuts')
#
# #plt.figure()
#
# ax1.plot(x_array, xlist, 'bo', markersize = 4, label = 'Horizontal cut')
# ax1.plot(x_array, gaussian_func(x_array, h_popt[0], h_popt[1], h_popt[2]), 'r', label='Best fit - Horizontal')
# ax1.legend()
#
# #plt.savefig('/home/time_user/TessC/fits_plots/Gauss_Horizontal_plot_stamp_11.png')
#
# ax2.plot(x_array, ylist, 'mo', markersize = 4, label = 'Vertical cut')
# ax2.plot(x_array, gaussian_func(x_array, v_popt[0], v_popt[1], v_popt[2]), 'k', label='Best fit- Vertical')
# ax2.legend()
#
# fig.tight_layout()
#
# plt.savefig('/home/time_user/TessC/fits_plots/Gauss_VandH_plot_stamp_11.png')
#
# print("For subgrid_stamp_11.png the Statistics for Horizontal cut are:")
#
# print("by curve_fit:")
# print("amplitude = ", round(h_popt[0],2))
# print("mean = ", round(h_popt[1],2))
# print("std = ", round(h_popt[2],2))
#
# print("by hand:")
# print("amp = ", round(max(xlist),2))
# print("mean = ", round(np.mean(xlist),2))
# print("std = ", round(np.std(xlist),2))
#
# print("For subgrid_stamp_11.png the Statistics for Vertical cut are:")
#
# print("by curve_fit:")
# print("amplitude = ", round(v_popt[0],2))
# print("mean = ", round(v_popt[1],2))
# print("std = ", round(v_popt[2],2))
#
# print("by hand:")
# print("amp = ", round(max(xlist),2))
# print("mean = ", round(np.mean(xlist),2))
# print("std = ", round(np.std(xlist),2))



files = glob.glob('/data/focus_sims/ciber_data/fits_files')
image_hdus = []

def gaussian_func(x, amp , mean, std):
    return amp*np.exp(-(x-mean)**2/(2*std**2))

#for f in range(5):
og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_13.FITS')# % int(f+11))
image_hdus.append(og_im[0].data)
image_data_filt = gaussian_filter(og_im[0].data, 5)
result = np.where(image_data_filt == np.amax(image_data_filt)) #returns indices
x_data = image_data_filt[:,result[0]] #result[0] is the single y data point
ylist = [item for sublist in x_data for item in sublist] #flattens lists to one list
x_array = np.arange(len(x_data)) #numbers from 0 to 327

popt, pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])

FWHM = 2*np.sqrt(2*np.log(2))*popt[2]

plt.figure()
plt.plot(x_array, ylist, 'go', markersize = 4, label = 'Image data')
plt.plot(x_array, gaussian_func(x_array, popt[0], popt[1], popt[2]), 'b', label='Best fit')
plt.axvspan(popt[1]-FWHM/2, popt[1]+FWHM/2, label= 'FWHM', facecolor= 'r', alpha=0.2)
plt.legend(loc="upper center", bbox_to_anchor=(0.8, 1))
# statistics.stdev(sample)
#print("For subgrid_stamp_11.png the Statistics are:")# % int(f+11))

#print("From curve_fit:")
# print("amplitude = ", round(popt[0],2))
#print("mean = ", round(popt[1],2))
print(popt)
print("std = ", round(popt[2],2))
print("FWHM = ", round(FWHM,2))


#print("From Data:")
# print("amp = ", round(max(ylist),2))
# print("mean = ", round(np.mean(ylist),2))
print("std = ", round(np.std(ylist),2))

#plt.savefig('/home/time_user/TessC/TEST_Gauss_plot_stamp_13.png')# % int(f+11))
