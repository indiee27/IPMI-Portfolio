# -*- coding: utf-8 -*-

#%%
# set MATPLOTLIB to use auto backend - this will display figures in separate windows and
# is required for the animation to display correctly
import matplotlib.pyplot as plt
%matplotlib auto

# Load the 2D lung MRI image using the imread function from the scikit-image python library
import skimage.io
img = skimage.io.imread('lung_MRI_slice.png')

# Check the data type of the image
print(img.dtype)

# convert data type to double to avoid errors when processing integers
import numpy as np

# convert img to double
img = np.double(img)
print(img.dtype)

# reorientate image to standard orientation
from PIL import ImageOps

def orientation_standard(img):
  # transpose - switch x and y dimensions
  transpose_image = np.transpose(img)
  # flip along second dimension - top to bottom
  flipped_image = np.flip(transpose_image)
  return flipped_image

source_img = orientation_standard(img)

# display the image using the dispImage function from utils.py
from utils import dispImage
plt.figure()
dispImage(img)
plt.figure()
dispImage(source_img)

#%%

# define a matrix representing a translation of [10, 20, 0]
T1 = np.matrix([[1, 0, 10],[0, 1, 20],[0, 0, 1]])
print(T1)

# inputs for deformation matrix
num_pix_x = len(source_img[0])
num_pix_y = len(source_img[1])

# create a deformation field from the affine matrix in utils.py
from utils import defFieldFromAffineMatrix

def_field = defFieldFromAffineMatrix(T1, num_pix_x, num_pix_y)

# resample the image using the deformation field
from utils import resampImageWithDefField

img_resampled = resampImageWithDefField(source_img, def_field, interp_method = 'linear', pad_value=np.NaN)

# display the transformed image
plt.figure()
dispImage(img_resampled)

# check the value of pixel 255,255 in the resampled image
print(img_resampled[255,255])

#%%

# resample using nearest neighbour and splinef2d
img_resampled_nearest_neighbour = resampImageWithDefField(source_img, def_field, interp_method = 'nearest', pad_value=np.NaN)
img_resampled_splinef2D = resampImageWithDefField(source_img, def_field, interp_method = 'splinef2d', pad_value=np.NaN)

# display in seperate figures
plt.figure()
dispImage(img_resampled_nearest_neighbour)
plt.figure()
dispImage(img_resampled_splinef2D)

# show difference images between new resamples and original linear
diff_linear_nearest = abs(img_resampled_nearest_neighbour - img_resampled)
plt.figure()
dispImage(diff_linear_nearest)

diff_linear_splinef2d = abs(img_resampled_splinef2D - img_resampled)
plt.figure()
dispImage(diff_linear_splinef2d)

#%%

# repeat the translation using a deformation matrix of [10.5, 20.5, 0]
T2 = np.matrix([[1, 0, 10.5],[0, 1, 20.5],[0, 0, 1]])

def_field_2 = defFieldFromAffineMatrix(T2, num_pix_x, num_pix_y)

img_resampled_2 = resampImageWithDefField(source_img, def_field_2, interp_method = 'linear', pad_value=np.NaN)

plt.figure()
dispImage(img_resampled_2)

# resample using nearest neighbour and splinef2D interpolation
img_resampled_2_nearest_neighbour = resampImageWithDefField(source_img, def_field_2, interp_method='nearest', pad_value=np.NaN)
plt.figure()
dispImage(img_resampled_2_nearest_neighbour)

img_resampled_2_splinef2d = resampImageWithDefField(source_img, def_field_2, interp_method = 'splinef2d', pad_value=np.NaN)
plt.figure()
dispImage(img_resampled_2_splinef2d)

# show the differences between new resamples and linear 2
diff_linear_2_nearest = abs(img_resampled_2_nearest_neighbour - img_resampled_2)
plt.figure()
dispImage(diff_linear_2_nearest)

diff_linear_2_splinef2d = abs(img_resampled_2_splinef2d - img_resampled_2)
plt.figure()
dispImage(diff_linear_2_splinef2d)

#%%

# difference images redisplayed with intensity limits of [-20, 20]
plt.figure()
dispImage(diff_linear_nearest, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_splinef2d, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_2_nearest, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_2_splinef2d, int_lims = [-20, 20])


#%%

# define a function to calculate the affine matrix for a rotation about a point

def affineMatrixForRotationAboutPoint(theta, p_coords):
  """
  function to calculate the affine matrix corresponding to an anticlockwise
  rotation about a point
  
  INPUTS:    theta: the angle of the rotation, specified in degrees
             p_coords: the 2D coordinates of the point that is the centre of
                 rotation. p_coords[0] is the x coordinate, p_coords[1] is
                 the y coordinate
  
  OUTPUTS:   aff_mat: a 3 x 3 affine matrix
  """
  # convert theta to radians
  theta = np.pi * theta/180
  
  # define matrices for translation and rotation
  T1 = np.matrix([[1, 0, -p_coords[0]],
                  [0, 1, -p_coords[1]],
                  [0,0,1]])
  
  T2 = np.matrix([[1, 0, p_coords[0]],
                  [0, 1, p_coords[1]],
                  [0,0,1]])
  
  R = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
  
  # compose matrix
  aff_mat = T2 * R * T1

  return aff_mat

#%%

# close any open figures
plt.close('all')

# USE THE ABOVE FUNCTION TO CALCULATE THE AFFINE MATRIX REPRESENTING AN ANTICLOCKWISE
# ROTATION OF 5 DEGREES ABOUT THE CENTRE OF THE IMAGE
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefField(source_img, rotation_transform_5_degree)

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

#%%

# APPLY THE SAME TRANSFORMATION AGAIN TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT. 
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES

for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefField(rotation_image_5_degree, rotation_transform)

dispImage(rotation_image)

#%%
# CHANGE THE CODE ABOVE SO THAT IT USES A PADDING VALUE OF 0 INSTEAD OF NAN

for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefField(rotation_image_5_degree, rotation_transform, pad_value=0)

dispImage(rotation_image)
  
#%%
  
# REPEAT THE ABOVE ANIMATION BUT USING NEAREST NEIGHBOUR
# FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY THE RESULT
# CREATE ROTATION
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefField(source_img, rotation_transform_5_degree, interp_method = 'nearest')

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

# DISPLAY RESULT USING SOURCE IMAGE INTENSITY LIMITS
plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

# THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefField(rotation_image_5_degree, rotation_transform, interp_method='nearest')

  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)

#%%
  
# REPEAT THE ABOVE ANIMATION BUT USING SPLINEF2D INTERPOLATION
# FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY THE RESULT
# CREATE ROTATION
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefField(source_img, rotation_transform_5_degree, interp_method = 'splinef2d')

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

# DISPLAY RESULT USING SOURCE IMAGE INTENSITY LIMITS
plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

# THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefField(rotation_image_5_degree, rotation_transform, interp_method='splinef2d')

  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)

#%%

# CREATE ANIMATIONS OF THE ROTATING IMAGES AS ABOVE, BUT COMPOSES THE
# TRANSFORMATIONS AT EACH STEP AND APPLIES THE COMPOSED TRANSFORMATION TO THE ORIGINAL IMAGE

# recreate the affine matrix corresponding to a rotation of 5 degrees about the centre of the image
R = affineMatrixForRotationAboutPoint(5, [(num_pix_x - 1)/2, (num_pix_y - 1)/2])
  
# create deformation field, resample original image, and display result
def_field = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
img_resampled = resampImageWithDefField(img, def_field)
dispImage(img_resampled, int_lims=int_lims_img)
plt.pause(0.05)

# create current matrix as copy of rotation matrix
R_current = R

for n in range(71):
  
  # COMPOSE CURRENT MATRIX AND ROTATION MATRIX
  R_current = R_current + R
  
  # CREATE DEFORMATION FIELD FOR CURRENT MATRIX, RESAMPLE ORIGINAL IMAGE, AND DISPLAY RESULT
  def_field = defFieldFromAffineMatrix(R_current, num_pix_x, num_pix_y)
  img_resampled = resampImageWithDefField(img, def_field)
  dispImage(img_resampled)
  
  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)

#%%

# REPEAT ALL PRIOR CODE TO USE PUSH INTERPOLATION INSTEAD OF PULL INTERPOLATION
from utils import resampImageWithDefFieldPushInterp

#%%

# define a matrix representing a translation of [10, 20, 0]
T1 = np.matrix([[1, 0, 10],[0, 1, 20],[0, 0, 1]])
print(T1)

# inputs for deformation matrix
num_pix_x = len(source_img[0])
num_pix_y = len(source_img[1])

# create a deformation field from the affine matrix in utils.py
from utils import defFieldFromAffineMatrix

def_field = defFieldFromAffineMatrix(T1, num_pix_x, num_pix_y)
img_resampled = resampImageWithDefFieldPushInterp(source_img, def_field, interp_method = 'linear', pad_value=np.NaN)

# display the transformed image
plt.figure()
dispImage(img_resampled)

# check the value of pixel 255,255 in the resampled image
print(img_resampled[255,255])

#%%

# resample using nearest neighbour and splinef2d
img_resampled_nearest_neighbour = resampImageWithDefFieldPushInterp(source_img, def_field, interp_method = 'nearest', pad_value=np.NaN)
img_resampled_splinef2D = resampImageWithDefFieldPushInterp(source_img, def_field, interp_method = 'splinef2d', pad_value=np.NaN)

# display in seperate figures
plt.figure()
dispImage(img_resampled_nearest_neighbour)
plt.figure()
dispImage(img_resampled_splinef2D)

# show difference images between new resamples and original linear
diff_linear_nearest = abs(img_resampled_nearest_neighbour - img_resampled)
plt.figure()
dispImage(diff_linear_nearest)

diff_linear_splinef2d = abs(img_resampled_splinef2D - img_resampled)
plt.figure()
dispImage(diff_linear_splinef2d)

#%%

# repeat the translation using a deformation matrix of [10.5, 20.5, 0]
T2 = np.matrix([[1, 0, 10.5],[0, 1, 20.5],[0, 0, 1]])

def_field_2 = defFieldFromAffineMatrix(T2, num_pix_x, num_pix_y)

img_resampled_2 = resampImageWithDefFieldPushInterp(source_img, def_field_2, interp_method = 'linear', pad_value=np.NaN)

plt.figure()
dispImage(img_resampled_2)

# resample using nearest neighbour and splinef2D interpolation
img_resampled_2_nearest_neighbour = resampImageWithDefFieldPushInterp(source_img, def_field_2, interp_method='nearest', pad_value=np.NaN)
plt.figure()
dispImage(img_resampled_2_nearest_neighbour)

img_resampled_2_splinef2d = resampImageWithDefFieldPushInterp(source_img, def_field_2, interp_method = 'splinef2d', pad_value=np.NaN)
plt.figure()
dispImage(img_resampled_2_splinef2d)

# show the differences between new resamples and linear 2
diff_linear_2_nearest = abs(img_resampled_2_nearest_neighbour - img_resampled_2)
plt.figure()
dispImage(diff_linear_2_nearest)

diff_linear_2_splinef2d = abs(img_resampled_2_splinef2d - img_resampled_2)
plt.figure()
dispImage(diff_linear_2_splinef2d)

#%%

# difference images redisplayed with intensity limits of [-20, 20]
plt.figure()
dispImage(diff_linear_nearest, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_splinef2d, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_2_nearest, int_lims = [-20, 20])

plt.figure()
dispImage(diff_linear_2_splinef2d, int_lims = [-20, 20])


#%%

# define a function to calculate the affine matrix for a rotation about a point

def affineMatrixForRotationAboutPoint(theta, p_coords):
  """
  function to calculate the affine matrix corresponding to an anticlockwise
  rotation about a point
  
  INPUTS:    theta: the angle of the rotation, specified in degrees
             p_coords: the 2D coordinates of the point that is the centre of
                 rotation. p_coords[0] is the x coordinate, p_coords[1] is
                 the y coordinate
  
  OUTPUTS:   aff_mat: a 3 x 3 affine matrix
  """
  # convert theta to radians
  theta = np.pi * theta/180
  
  # define matrices for translation and rotation
  T1 = np.matrix([[1, 0, -p_coords[0]],
                  [0, 1, -p_coords[1]],
                  [0,0,1]])
  
  T2 = np.matrix([[1, 0, p_coords[0]],
                  [0, 1, p_coords[1]],
                  [0,0,1]])
  
  R = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
  
  # compose matrix
  aff_mat = T2 * R * T1

  return aff_mat

#%%

# close any open figures
plt.close('all')

# USE THE ABOVE FUNCTION TO CALCULATE THE AFFINE MATRIX REPRESENTING AN ANTICLOCKWISE
# ROTATION OF 5 DEGREES ABOUT THE CENTRE OF THE IMAGE
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefFieldPushInterp(source_img, rotation_transform_5_degree)

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

#%%

# APPLY THE SAME TRANSFORMATION AGAIN TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT. 
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES

for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefFieldPushInterp(rotation_image_5_degree, rotation_transform)

dispImage(rotation_image)
  
#%%
  
# REPEAT THE ABOVE ANIMATION BUT USING NEAREST NEIGHBOUR
# FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY THE RESULT

# CREATE ROTATION
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefFieldPushInterp(source_img, rotation_transform_5_degree, interp_method = 'nearest')

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

# DISPLAY RESULT USING SOURCE IMAGE INTENSITY LIMITS
plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

# THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefFieldPushInterp(rotation_image_5_degree, rotation_transform, interp_method='nearest')

  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)

#%%
  
# REPEAT THE ABOVE ANIMATION BUT USING SPLINEF2D INTERPOLATION

# FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY THE RESULT
# CREATE ROTATION
R = affineMatrixForRotationAboutPoint(5, [0,0])

# USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE
rotation_transform_5_degree = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
rotation_image_5_degree = resampImageWithDefFieldPushInterp(source_img, rotation_transform_5_degree, interp_method = 'splinef2d')

# DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure(1)
dispImage(rotation_image_5_degree)

# DISPLAY RESULT USING SOURCE IMAGE INTENSITY LIMITS
plt.figure(2)
smallest = np.amin(source_img)
biggest = np.amax(source_img)
int_lims_img = [smallest, biggest]
dispImage(rotation_image_5_degree, int_lims = int_lims_img) 

# THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
# REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n in range(71):
  R = affineMatrixForRotationAboutPoint((n*5), [0,0])
  rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
  rotation_image = resampImageWithDefFieldPushInterp(rotation_image_5_degree, rotation_transform, interp_method='splinef2d')

  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)


#%%

# CREATE ANIMATIONS OF THE ROTATING IMAGES AS ABOVE, BUT COMPOSES THE
# TRANSFORMATIONS AT EACH STEP AND APPLIES THE COMPOSED TRANSFORMATION TO THE ORIGINAL IMAGE

# recreate the affine matrix corresponding to a rotation of 5 degrees about the centre of the image
R = affineMatrixForRotationAboutPoint(5, [(num_pix_x - 1)/2, (num_pix_y - 1)/2])
  
# create deformation field, resample original image, and display result
def_field = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y)
img_resampled = resampImageWithDefFieldPushInterp(img, def_field)
dispImage(img_resampled, int_lims=int_lims_img)
plt.pause(0.05)

# create current matrix as copy of rotation matrix
R_current = R

for n in range(71):
  
  # COMPOSE CURRENT MATRIX AND ROTATION MATRIX
  R_current = R_current + R
  
  # CREATE DEFORMATION FIELD FOR CURRENT MATRIX, RESAMPLE ORIGINAL IMAGE, AND DISPLAY RESULT
  def_field = defFieldFromAffineMatrix(R_current, num_pix_x, num_pix_y)
  img_resampled = resampImageWithDefFieldPushInterp(img, def_field)
  dispImage(img_resampled)
  
  #add a short pause after displaying the image so the figure display updates
  plt.pause(0.05)
