# -*- coding: utf-8 -*-

#%%
# set MATPLOTLIB to use auto backend - this will display figures in separate windows and
# is required for the animation to display correctly
import matplotlib.pyplot as plt
%matplotlib qt

#%%
# Load in the three 2D images required for these exercises using the imread function from the
# scikit-image python library
import skimage.io
ct_img_int8 = skimage.io.imread('ct_slice_int8.png')
ct_img_int16 = skimage.io.imread('ct_slice_int16.png')
mr_img_int16 = skimage.io.imread('mr_slice_int16.png')

# data type of each image
print(ct_img_int8.dtype)
print(ct_img_int16.dtype)
print(mr_img_int16.dtype)

#%%
import numpy as np
# convert images to double precision
ct_img_8 = np.double(ct_img_int8)
ct_img_16 = np.double(ct_img_int16)
mr_img = np.double(mr_img_int16)

# convert to standard orientation
from PIL import ImageOps

def orientation_standard(img):
  # transpose - switch x and y dimensions
  transpose_image = np.transpose(img)
  # flip along second dimension - top to bottom
  flipped_image = np.flip(transpose_image)
  return flipped_image

ct_source_img_8 = orientation_standard(ct_img_8)
ct_source_img_16 = orientation_standard(ct_img_16)
mr_source_img = orientation_standard(mr_img)

# display each image using the dispimage function from utils
from utils2 import dispImage
plt.figure(1)
dispImage(ct_source_img_8)
plt.figure(2)
dispImage(ct_source_img_16)
plt.figure(3)
dispImage(mr_source_img)

#%%
# ***************
# EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES ARE ALL
# ROTATED BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND DISPLAY THE
# TRANSFORMED 8 BIT CT IMAGE
# ***************
from utils2 import affineMatrixForRotationAboutPoint
from utils2 import defFieldFromAffineMatrix
from utils2 import resampImageWithDefField
from utils2 import calcSSD
theta = np.arange(-90, 91, 1)
SSDs = np.zeros((theta.size, 4))
num_pix_x, num_pix_y = ct_img_int8.shape
for n in range(theta.size):
  
  # CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  aff_mat = 
  def_field = 
  # RESAMPLE THE IMAGES
  ct_img_int8_resamp = 
  ct_img_int16_resamp = 
  mr_img_int16_resamp = 
  
  # DISPLAY THE TRANSFORMED 8 BIT CT IMAGE
  
  #add a short pause so the figure display updates
  plt.pause(0.05)
  
  # ***************
  # UNCOMMENT AND EDIT THE CODE BELOW TO CALCULATE THE SSD VALUES BETWEEN:
  # 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
  # 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
  # 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
  # 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
  #SSDs[n, 0] = 
  #SSDs[n, 1] = 
  #SSDs[n, 2] = 
  #SSDs[n, 3] = 
  # ***************
  
#%%
# plot the SSD values (y axis) against the angle theta (x axis)
plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, SSDs[:, 0])
plt.title('SSD: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, SSDs[:, 1])
plt.title('SSD:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, SSDs[:, 2])
plt.title('SSD:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, SSDs[:, 3])
plt.title('SSD:  8-bit CT and  8-bit CT')

#%%
# ***************
# EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES ARE ALL ROTATED
# BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND THE MSD IS CALCULATED BETWEEN:
# 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
# 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
# 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
# 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
from utils2 import calcMSD
MSDs = np.zeros((theta.size, 4))
for n in range(theta.size):
  
  # CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  aff_mat = 
  def_field = 
  # RESAMPLE THE IMAGES
  ct_img_int8_resamp = 
  ct_img_int16_resamp = 
  mr_img_int16_resamp = 
  
  # CALCULATE THE MSD VALUES
  MSDs[n, 0] = 
  MSDs[n, 1] = 
  MSDs[n, 2] = 
  MSDs[n, 3] = 
  
# plot the MSD values (y axis) against the angle theta (x axis)
plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, MSDs[:, 0])
plt.title('MSD: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, MSDs[:, 1])
plt.title('MSD:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, MSDs[:, 2])
plt.title('MSD:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, MSDs[:, 3])
plt.title('MSD:  8-bit CT and  8-bit CT')
# ***************

#%%
# ***************
# EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES ARE ALL ROTATED
# BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND THE NORMALISED CROSS CORRELATION
# AND THE JOINT AND MARGINAL ENTROPIES ARE CALCULATED BETWEEN:
# 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
# 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
# 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
# 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
from utils2 import calcNCC
from utils2 import calcEntropies
NCCs = np.zeros((theta.size, 4))
H_ABs = np.zeros((theta.size, 4))
H_As = np.zeros((theta.size, 4))
H_Bs = np.zeros((theta.size, 4))
for n in range(theta.size):
  
  # CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  aff_mat = 
  def_field = 
  # RESAMPLE THE IMAGES
  ct_img_int8_resamp = 
  ct_img_int16_resamp = 
  mr_img_int16_resamp = 
  
  # CALCULATE THE NCC VALUES
  NCCs[n, 0] = 
  NCCs[n, 1] = 
  NCCs[n, 2] = 
  NCCs[n, 3] = 
  
  # CALCULATE THE JOINT AND MARGINAL ENTROPIES
  H_ABs[n, 0], H_As[n, 0], H_Bs[n, 0]  = 
  H_ABs[n, 1], H_As[n, 1], H_Bs[n, 1]  = 
  H_ABs[n, 2], H_As[n, 2], H_Bs[n, 2]  = 
  H_ABs[n, 3], H_As[n, 3], H_Bs[n, 3]  = 
# ***************  
  
# ***************
# EDIT THE CODE BELOW TO CALCULATE THE MUTUAL INFORMATION AND NORMALISED MUTUAL INFORMATION
# (note - this code is outside of the for loop)
MIs = 
NMIs = 
# *************** 
  
# plot the results for NCC, H_AB, MI, and NMI in separate figures
plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, NCCs[:, 0])
plt.title('NCC: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, NCCs[:, 1])
plt.title('NCC:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, NCCs[:, 2])
plt.title('NCC:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, NCCs[:, 3])
plt.title('NCC:  8-bit CT and  8-bit CT')

plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, H_ABs[:, 0])
plt.title('H_AB: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, H_ABs[:, 1])
plt.title('H_AB:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, H_ABs[:, 2])
plt.title('H_AB:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, H_ABs[:, 3])
plt.title('H_AB:  8-bit CT and  8-bit CT')

plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, MIs[:, 0])
plt.title('MI: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, MIs[:, 1])
plt.title('MI:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, MIs[:, 2])
plt.title('MI:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, MIs[:, 3])
plt.title('MI:  8-bit CT and  8-bit CT')

plt.figure()
plt.subplot(2,2,1)
plt.plot(theta, NMIs[:, 0])
plt.title('NMI: 16-bit CT and 16-bit CT')
plt.subplot(2,2,2)
plt.plot(theta, NMIs[:, 1])
plt.title('NMI:  16-bit CT and  8-bit CT')
plt.subplot(2,2,3)
plt.plot(theta, NMIs[:, 2])
plt.title('NMI:  16-bit CT and 16-bit MR')
plt.subplot(2,2,4)
plt.plot(theta, NMIs[:, 3])
plt.title('NMI:  8-bit CT and  8-bit CT')
