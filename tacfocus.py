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



''' ############ Displays only one FITS image ###############'''
# image_file = get_pkg_data_filename('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_04.FITS')
#
# image_data = fits.getdata(image_file, ext=0) #stores data as 2D array
# image_data_int = image_data.astype(int)
# print(type(image_data))
# print(fits.info(image_file))
# print(image_data.shape) #Shape of ciber files are is 328 by 328

# plt.figure()
# plt.imshow(image_data, cmap='gray')
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

'''
Horizontal and vertical cut of data for all blob images
'''
#
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
#     result = np.where(image_data_filt == np.amax(image_data_filt))
#     #print(result) #x and y coords of max
#     x_data = image_data_filt[:,result[1]]
#     xlist = [item for sublist in x_data for item in sublist] #flattens list into one list
#
#     y_data = image_data_filt[result[0],:]
#     ylist = [item for sublist in x_data for item in sublist]
#     x_array = np.arange(len(x_data)) #numbers from 0 to 327 for x-axis of guassian plot
#
#     h_popt, h_pcov = curve_fit(gaussian_func, x_array, xlist, p0 = [max(xlist), np.mean(xlist), np.std(xlist)])
#     v_popt, v_pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])
#
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('H and V data cuts stamp_%.2d' % int(f+11))
#
#     #plt.figure()
#     ax1.plot(x_array, xlist, 'bo', markersize = 4, label = 'Horizontal cut')
#     ax1.plot(x_array, gaussian_func(x_array, h_popt[0], h_popt[1], h_popt[2]), 'r', label='Best fit - Horizontal')
#     ax1.legend()
#
#     #plt.savefig('/home/time_user/TessC/fits_plots/Gauss_Horizontal_plot_stamp_%.2d.png' % int(f+11))
#
#     ax2.plot(x_array, ylist, 'mo', markersize = 4, label = 'Vertical cut')
#     ax2.plot(x_array, gaussian_func(x_array, v_popt[0], v_popt[1], v_popt[2]), 'k', label='Best fit- Vertical')
#     ax2.legend()
#
#     fig.tight_layout()
#
#     plt.savefig('/home/time_user/TessC/fits_plots/Gauss_VandH_plot_stamp_%.2d.png' % int(f+11))
#
#     print("For subgrid_stamp_%.2d.png the Statistics for Horizontal cut are:" % int(f+11))
#
#     print("by curve_fit:")
#     print("amplitude = ", round(h_popt[0],2))
#     print("mean = ", round(h_popt[1],2))
#     print("std = ", round(h_popt[2],2))
#
#     print("by hand:")
#     print("amp = ", round(max(xlist),2))
#     print("mean = ", round(np.mean(xlist),2))
#     print("std = ", round(np.std(xlist),2))
#
#     print("For subgrid_stamp_%.2d.png the Statistics for Vertical cut are:" % int(f+11))
#
#     print("by curve_fit:")
#     print("amplitude = ", round(v_popt[0],2))
#     print("mean = ", round(v_popt[1],2))
#     print("std = ", round(v_popt[2],2))
#
#     print("by hand:")
#     print("amp = ", round(max(xlist),2))
#     print("mean = ", round(np.mean(xlist),2))
#     print("std = ", round(np.std(xlist),2))

'''
Horizontal and vertical gaussian data for all blob images with FWHM:
'''
#
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
#     result = np.where(image_data_filt == np.amax(image_data_filt))
#     #print(result) #x and y coords of max
#     x_data = image_data_filt[:,result[1]]
#     xlist = [item for sublist in x_data for item in sublist] #flattens list into one list
#
#     y_data = image_data_filt[result[0],:]
#     ylist = [item for sublist in x_data for item in sublist]
#     x_array = np.arange(len(x_data)) #numbers from 0 to 327 for x-axis of guassian plot
#
#     h_popt, h_pcov = curve_fit(gaussian_func, x_array, xlist, p0 = [max(xlist), np.mean(xlist), np.std(xlist)])
#     h_FWHM = 2*np.sqrt(2*np.log(2))*h_popt[2]
#
#     v_popt, v_pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])
#     v_FWHM = 2*np.sqrt(2*np.log(2))*v_popt[2]
#
#
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('H and V data cuts stamp_%.2d' % int(f+11))
#
#     #plt.figure()
#     ax1.plot(x_array, xlist, 'bo', markersize = 4, label = 'Horizontal cut')
#     ax1.plot(x_array, gaussian_func(x_array, h_popt[0], h_popt[1], h_popt[2]), 'r', label='Best fit - Horizontal')
#     ax1.axvspan(h_popt[1]-h_FWHM/2, h_popt[1]+h_FWHM/2, label= 'FWHM', facecolor= 'r', alpha=0.2)
#     ax1.legend()
#
#     #plt.savefig('/home/time_user/TessC/fits_plots/Gauss_Horizontal_plot_stamp_%.2d.png' % int(f+11))
#
#     ax2.plot(x_array, ylist, 'mo', markersize = 4, label = 'Vertical cut')
#     ax2.plot(x_array, gaussian_func(x_array, v_popt[0], v_popt[1], v_popt[2]), 'k', label='Best fit- Vertical')
#     ax2.axvspan(v_popt[1]-v_FWHM/2, v_popt[1]+v_FWHM/2, label= 'FWHM', facecolor= 'r', alpha=0.2)
#     ax2.legend()
#
#
#     fig.tight_layout()
#
#     plt.savefig('/home/time_user/TessC/Gauss_VandH_plot_stamp_%.2d.png' % int(f+11))
#
#     print("For subgrid_stamp_%.2d.png :" % int(f+11))
#     print("FWHM for Horizontal = ", round(h_FWHM,2))
#     print("FWHM for Vertical = ", round(v_FWHM,2))


'''
Troubleshooting.... Why is FWHM negative?
'''

files = glob.glob('/data/focus_sims/ciber_data/fits_files')
image_hdus = []

def gaussian_func(x, amp , mean, std):
    return amp*np.exp(-(x-mean)**2/(2*std**2))

#for f in range(5):
og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_11.FITS')# % int(f+11))
image_hdus.append(og_im[0].data)
image_data_filt = gaussian_filter(og_im[0].data, 5)
result = np.where(image_data_filt == np.amax(image_data_filt))
#print(result) #x and y coords of max
x_data = image_data_filt[:,result[1]]
xlist = [item for sublist in x_data for item in sublist] #flattens list into one list

y_data = image_data_filt[result[0],:]
ylist = [item for sublist in x_data for item in sublist]
x_array = np.arange(len(x_data)) #numbers from 0 to 327 for x-axis of guassian plot

h_popt, h_pcov = curve_fit(gaussian_func, x_array, xlist, p0 = [max(xlist), np.mean(xlist), np.std(xlist)])
h_FWHM = 2*np.sqrt(2*np.log(2))*h_popt[2]

v_popt, v_pcov = curve_fit(gaussian_func, x_array, ylist, p0 = [max(ylist), np.mean(ylist), np.std(ylist)])
v_FWHM = 2*np.sqrt(2*np.log(2))*v_popt[2]


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('H and V data cuts stamp_11')# % int(f+11))

#plt.figure()
ax1.plot(x_array, xlist, 'bo', markersize = 4, label = 'Horizontal cut')
ax1.plot(x_array, gaussian_func(x_array, h_popt[0], h_popt[1], h_popt[2]), 'r', label='Best fit - Horizontal')
ax1.axvspan(h_popt[1]-h_FWHM/2, h_popt[1]+h_FWHM/2, label= 'FWHM', facecolor= 'r', alpha=0.2)
ax1.legend(loc="upper center", bbox_to_anchor=(0.8, 1))

#plt.savefig('/home/time_user/TessC/fits_plots/Gauss_Horizontal_plot_stamp_%.2d.png' % int(f+11))

ax2.plot(x_array, ylist, 'mo', markersize = 4, label = 'Vertical cut')
ax2.plot(x_array, gaussian_func(x_array, v_popt[0], v_popt[1], v_popt[2]), 'k', label='Best fit- Vertical')
ax2.axvspan(v_popt[1]-v_FWHM/2, v_popt[1]+v_FWHM/2, label= 'FWHM', facecolor= 'r', alpha=0.2)
ax2.legend(loc="upper center", bbox_to_anchor=(0.8, 1))

fig.tight_layout()

plt.savefig('/home/time_user/TessC/Gauss_VandH_plot_stamp_11.png')# % int(f+11))

print("For subgrid_stamp_11.png :")# % int(f+11))
print("FWHM for Horizontal = ", round(h_FWHM,2))
print("FWHM for Vertical = ", round(v_FWHM,2))
