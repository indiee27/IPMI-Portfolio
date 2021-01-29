%{
template script for use in the image registration exercises 2 for module MPHY0025 (IPMI)

Jamie McClelland
UCL
%}
%%
% clear variables from memory and close all figures
clear
close all

%%
% Load in the three 2D images required for these exercises using the imread function
ct_img_int8 = imread('ct_slice_int8.png');
ct_img_int16 = imread('ct_slice_int16.png');
mr_img_int16 = imread('mr_slice_int16.png');

% ***************
% ADD CODE HERE TO DISPLAY THE DATA TYPE OF EACH IMAGE
% ***************

%%
% ***************
% ADD CODE HERE TO CONVERT ALL THE IMAGES TO DOUBLE DATA TYPE AND TO REORIENTATE THEM
% INTO 'STANDARD ORIENTATION'
% ***************

% ***************
% ADD CODE HERE TO DISPLAY EACH IMAGE IN A SEPARATE FIGURE
%
% USE THE FIGURES TO INSPECT THE INTENSITY VALUES IN EACH OF THE IMAGES
% ***************

%%
% ***************
% UNCOMMENT AND EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES
% ARE ALL ROTATED BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND DISPLAY THE
% TRANSFORMED 8 BIT CT IMAGE
% ***************
theta = -90:90;
SSDs = zeros(length(theta), 4);
[num_pix_x, num_pix_y] = size(ct_img_int8);
for n = 1:length(theta)
  
  % CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  %aff_mat = 
  %def_field = 
  % RESAMPLE THE IMAGES
  %ct_img_int8_resamp = 
  %ct_img_int16_resamp = 
  %mr_img_int16_resamp = 
  
  % DISPLAY THE TRANSFORMED 8 BIT CT IMAGE
  
  %add a short pause so the figure display updates
  pause(0.05);
  
  % ***************
  % UNCOMMENT AND EDIT THE CODE BELOW TO CALCULATE THE SSD VALUES BETWEEN:
  % 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
  % 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
  % 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
  % 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
  %SSDs(n, 0) = 
  %SSDs(n, 1) = 
  %SSDs(n, 2) = 
  %SSDs(n, 3) = 
  % ***************
end

%%
% plot the SSD values (y axis) against the angle theta (x axis)
figure();
subplot(2,2,1);
plot(theta, SSDs(:, 1));
title('SSD: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, SSDs(:, 2));
title('SSD:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, SSDs(:, 3));
title('SSD:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, SSDs(:, 4));
title('SSD:  8-bit CT and  8-bit CT');

%%
% ***************
% UNCOMMENT AND EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES
% ARE ALL ROTATED BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND THE MSD IS
% CALCULATED BETWEEN:
% 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
% 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
% 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
% 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
MSDs = zeros(length(theta), 4);
for n = 1:length(theta)
  
  % CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  %aff_mat = 
  %def_field = 
  % RESAMPLE THE IMAGES
  %ct_img_int8_resamp = 
  %ct_img_int16_resamp = 
  %mr_img_int16_resamp = 
  
  % CALCULATE THE MSD VALUES
  %MSDs(n, 1) = 
  %MSDs(n, 2) = 
  %MSDs(n, 3) = 
  %MSDs(n, 4) = 
  
end

% plot the MSD values (y axis) against the angle theta (x axis)
figure();
subplot(2,2,1);
plot(theta, MSDs(:, 1));
title('MSD: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, MSDs(:, 2));
title('MSD:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, MSDs(:, 3));
title('MSD:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, MSDs(:, 4));
title('MSD:  8-bit CT and  8-bit CT');
% ***************

%%
% ***************
% UNCOMMENT AND EDIT THE CODE BELOW SO THAT ON EACH ITERATION OF THE FOR LOOP THE IMAGES
% ARE ALL ROTATED BY THE CURRENT VALUE OF THETA ABOUT THE POINT 10,10 AND THE NORMALISED
% CROSS CORRELATION AND THE JOINT AND MARGINAL ENTROPIES ARE CALCULATED BETWEEN:
% 1) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 16 BIT CT IMAGE
% 2) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
% 3) THE ORIGINAL 16 BIT CT IMAGE AND THE TRANSFORMED MR IMAGE
% 4) THE ORIGINAL 8 BIT CT IMAGE AND THE TRANSFORMED 8 BIT CT IMAGE
NCCs = zeros(length(theta), 4);
H_ABs = zeros(length(theta), 4);
H_As = zeros(length(theta), 4);
H_Bs = zeros(length(theta), 4);
for n = 1:length(theta)
  
  % CREATE AFFINE MATRIX AND CORRESPONDING DEFORMATION FIELD
  %aff_mat = 
  %def_field = 
  % RESAMPLE THE IMAGES
  %ct_img_int8_resamp = 
  %ct_img_int16_resamp = 
  %mr_img_int16_resamp = 
  
  % CALCULATE THE NCC VALUES
  %NCCs(n, 1) = 
  %NCCs(n, 2) = 
  %NCCs(n, 3) = 
  %NCCs(n, 4) = 
  
  % CALCULATE THE JOINT AND MARGINAL ENTROPIES
  %[H_ABs(n, 1), H_As(n, 1), H_Bs(n, 1)]  = 
  %[H_ABs(n, 2), H_As(n, 2), H_Bs(n, 2)]  = 
  %[H_ABs(n, 3), H_As(n, 3), H_Bs(n, 3)]  = 
  %[H_ABs(n, 4), H_As(n, 4), H_Bs(n, 4)]  = 
% ***************
end
  
% ***************
% UNCOMMENT AND EDIT THE CODE BELOW TO CALCULATE THE MUTUAL INFORMATION AND NORMALISED
% MUTUAL INFORMATION
%MIs = 
%NMIs = 
% *************** 
  
% plot the results for NCC, H_AB, MI, and NMI in separate figures
figure();
subplot(2,2,1);
plot(theta, NCCs(:, 1));
title('NCC: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, NCCs(:, 2));
title('NCC:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, NCCs(:, 3));
title('NCC:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, NCCs(:, 4));
title('NCC:  8-bit CT and  8-bit CT');

figure();
subplot(2,2,1);
plot(theta, H_ABs(:, 1));
title('H\_AB: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, H_ABs(:, 2));
title('H\_AB:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, H_ABs(:, 3));
title('H\_AB:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, H_ABs(:, 4));
title('H\_AB:  8-bit CT and  8-bit CT');

figure();
subplot(2,2,1);
plot(theta, MIs(:, 1));
title('MI: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, MIs(:, 2));
title('MI:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, MIs(:, 3));
title('MI:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, MIs(:, 4));
title('MI:  8-bit CT and  8-bit CT');

figure();
subplot(2,2,1);
plot(theta, NMIs(:, 1));
title('NMI: 16-bit CT and 16-bit CT');
subplot(2,2,2);
plot(theta, NMIs(:, 2));
title('NMI:  16-bit CT and  8-bit CT');
subplot(2,2,3);
plot(theta, NMIs(:, 3));
title('NMI:  16-bit CT and 16-bit MR');
subplot(2,2,4);
plot(theta, NMIs(:, 4));
title('NMI:  8-bit CT and  8-bit CT');
