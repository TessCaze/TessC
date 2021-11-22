import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import statistics
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from astropy.io import fits
import glob
from scipy import stats

#fits.info(image_file)
#print(type(image_data))
#print(fits.info(image_file))
#print(image_data.shape) #Shape of ciber files are is 328 by 328


''' ############ Displays only one FITS file ###############'''
# image_file = get_pkg_data_filename('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_04.FITS')
# image_data = fits.getdata(image_file, ext=0) #stores data as 2D array
# image_data_int = image_data.astype(int)
#
# plt.figure()
# plt.imshow(image_data, cmap='gray')
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.savefig('fits_images/fits_image_1.png')

''' ############ Displays each of the fits files in a separate folder ###############'''
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
# plt.savefig('/data/focus_sims/ciber_data/fits_images/all_fits_images.png')


'''
Here I am testing code with multiple images cutting a row from the xaxis and plot it with a Gaussian filter
'''
files = glob.glob('/data/focus_sims/ciber_data/fits_files')
image_hdus = []

for f in range(5):
    og_im = fits.open('/data/focus_sims/ciber_data/fits_files/subgrid_stamp_%.2d.FITS' % int(f+11))
    image_hdus.append(og_im[0].data)
    image_data_filt = gaussian_filter(og_im[0].data, 5)
    result = np.where(image_data_filt == np.amax(image_data_filt)) #returns indices
    #print(image_data_filt[:,result[0]]) #lists of lists
    plt.figure()
    plt.imshow(image_data_filt, cmap='gray')
    x_data = image_data_filt[:,result[0]]
    plt.plot(x_data)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.axhline(y=result[0], color='r', linestyle='-')
    plt.savefig('/data/focus_sims/ciber_data/fits_plots/guassian_subgrid_%.2d.png' % int(f+11))
