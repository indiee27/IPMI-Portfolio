%% Set up
clear
close all

% load images
ct_img_int8 = imread('ct_slice_int8.png');
ct_img_int16 = imread('ct_slice_int16.png');
mr_img_int16 = imread('mr_slice_int16.png');

% display data type of each image
disp(class(ct_img_int8))
disp(class(ct_img_int16))
disp(class(mr_img_int16))

% double data type
ct_img_8 = double(ct_img_int8);
ct_img_16 = double(ct_img_int16);
mr_img_16 = double(mr_img_int16);


% reorientate images
ct_source_8 = orientation_standard(ct_img_8);
ct_source_16 = orientation_standard(ct_img_16);
mr_source_16 = orientation_standard(mr_img_16);

dispImage(ct_source_8)

% function to reorientate images to standard
function orientation_standard(img)
img2 = transpose(img);
img3 = flip(img2,2);
end

