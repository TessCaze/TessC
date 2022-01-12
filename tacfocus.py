import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import statistics
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from astropy.io import fits
import glob
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

#fits.info(image_file)
#print(type(image_data))
#print(fits.info(image_file))
#print(image_data.shape) #Shape of ciber files are is 328 by 328

''' ############ Displays only one FITS image ###############'''
# image_file = get_pkg_data_filename('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_04.FITS')
# image_data = fits.getdata(image_file, ext=0) #stores data as 2D array
# image_data_int = image_data.astype(int)
#
# plt.figure()
# plt.imshow(image_data, cmap='gray')
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.savefig('fits_images/fits_image_1.png')

''' ############ Displays each of the fits images in a separately ###############'''
# files = glob.glob('/data/focus_sims/ciber_data/fits_files')
#
# image_hdus = []
# for f in range(18):
#     og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_%.2d.FITS' % int(f+4))
#     image_hdus.append(og_im[0].data)
#     plt.figure()
#     plt.imshow(og_im[0].data, cmap='gray')
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.savefig('/data/focus_sims/ciber_data/fits_images/subgrid_stamp_%.2d.png' % int(f+4))
#

''' ############ Displays all the fits images together ###############'''
# files = glob.glob('/data/focus_sims/ciber_data/fits_files')
#
# fig = plt.figure(figsize=(15, 15))
# fig.suptitle('Images of Ciber Data', fontsize=40)
# columns = 6
# rows = 3
#
# image_hdus = []
# for f in range(18):
#     og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_%.2d.FITS' % int(f+4))
#     image_hdus.append(og_im[0].data)
#     ax = fig.add_subplot(rows, columns, f+1)
#     ax.set_title('subgrid_stamp_%.2d' % int(f+4))
#     plt.imshow(og_im[0].data, cmap='gray')
#     plt.gca().invert_yaxis()
#
# plt.tight_layout()
# plt.savefig('/home/time_user/TessC/fits_images/all_fits_images.png')


'''
This Code takes a horizontal cut out of the fits images and plots it fitting a gaussian function
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
#     print("For subgrid_stamp_%.2d.png the statistics are:" % int(f+11))
#     print("amplitude = ", round(popt[0],2))
#     print("mean = ", round(popt[1],2))
#     print("std = ", round(popt[2],2))
# 
#     plt.savefig('/home/time_user/TessC/fits_plots/Gauss_plot_stamp_%.2d.png' % int(f+11))

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
#     popt, pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])
#
#     plt.figure()
#     plt.plot(x_array, ylist, 'go', markersize = 4, label = 'Image data')
#     plt.plot(x_array, gaussian_func(x_array, popt[0], popt[1], popt[2]), 'b', label='Best fit')
#     plt.legend()
#     # statistics.stdev(sample)
#     print("For subgrid_stamp_%.2d.png the Statistics are:" % int(f+11))
#
#     print("by curve_fit:")
#     print("amplitude = ", round(popt[0],2))
#     print("mean = ", round(popt[1],2))
#     print("std = ", round(popt[2],2))
#
#     print("by hand:")
#     print("amp = ", round(max(ylist),2))
#     print("mean = ", round(np.mean(ylist),2))
#     print("std = ", round(np.std(ylist),2))
#
#     plt.savefig('/home/time_user/TessC/fits_plots/Gauss_plot_stamp_%.2d.png' % int(f+11))
