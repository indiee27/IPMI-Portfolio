# jupyter script made by Indie 10th march
# -*- coding: utf -8 -*-

#%%
## SET MATPLOTLIB TO BACKEND 

import matplotlib as plt 
%matplotlib qt

#%%
## IMPORTS AND IMAGE CORRECTIONS

# import functions
from demonsReg import demonsReg
from utils3 import calcJacobian, dispImage

# import libraries
import skimage.io
import numpy as np

# import images
source_MR_1 = skimage.io.imread('cine_MR_1.png')
source_MR_2 = skimage.io.imread('cine_MR_2.png')
source_MR_3 = skimage.io.imread('cine_MR_3.png')

# convert to doubles
double_MR_1 = np.double(source_MR_1)
double_MR_2 = np.double(source_MR_2)
double_MR_3 = np.double(source_MR_3)

# standard orientation function
def orientation_standard(img):
    transpose_image = np.transpose(img)
    flipped_image = np.flip(transpose_image)
    return flipped_image

# standard orientation images
MR_1 = orientation_standard(double_MR_1)
MR_2 = orientation_standard(double_MR_2)
MR_3 = orientation_standard(double_MR_3)

#%%
## MR1 VS MR2 at default values

demonsReg(MR_1,MR_2)

#%%
## MR1 vs MR2, sigma_elastic set to zero

demonsReg(MR_1, MR_2, sigma_elastic=0)

#%%

demonsReg(MR_1, MR_2, sigma_fluid=0)

#%%

demonsReg(MR_1, MR_2, num_lev=1)

#%%

demonsReg(MR_1, MR_2, num_lev=6)
calcJacobian(def_field)
dispImage(J_Mat)

#%%

demonsReg(MR_1, MR_2, sigma_elastic = 0.5, sigma_fluid = 1, num_lev = 3, use_composition = True)
