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
from scipy import stats
from scipy.ndimage import gaussian_filter

#fits.info(image_file)
#print(type(image_data))
#print(fits.info(image_file))
#print(image_data.shape) #Shape of ciber files are is 328 by 328
#fits.info(image_file)
#print(type(image_data))
#print(fits.info(image_file))
#print(image_data.shape) #Shape of ciber files are is 328 by 328


''' ############ Displays only one FITS file ###############'''
image_file = get_pkg_data_filename('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_04.FITS')
image_data = fits.getdata(image_file, ext=0) #stores data as 2D array
image_data_int = image_data.astype(int)

plt.figure()
plt.imshow(image_data, cmap='gray')
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig('tac_fits_images/fits_image_1.png')

''' ############ gaussian_filter ###############'''
#
# image_file = get_pkg_data_filename('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_12.FITS')
# image_data = fits.getdata(image_file, ext=0) #stores data as 2D array
# image_data_int = image_data.astype(int)
#
# #scipy.ndimage.gaussian_filter
# #out_im = scipy.ndimage.gaussian_filter(input, [3,3], order=0, mode='constant', cval=0.0, truncate=4.0)
# image_data_filt = scipy.ndimage.gaussian_filter(image_data, [3,3], order=0, mode='constant', cval=0.0, truncate=4.0)
#
# plt.figure()
# plt.imshow(image_data_filt, cmap='gray')
# plt.gca().invert_yaxis()
# plt.colorbar()
#
#
# #plt.savefig('fits_plots/plot.png')
# plt.savefig('tac_fits_plots/plot_stamp_12.png')
