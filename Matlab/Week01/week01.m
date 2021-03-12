%% Week 1 script

%% admin stuff
clear
close all

%% loading image
img = imread('lung_MRI_slice.png');

% check the data type of the image
disp(class(img));

% convert data type to double
img = double(img);
disp(class(img));

% flip image to standard rotation
img = transpose(img);
img = flip(img,2);
source_img = img;

% display source image
dispImage(source_img)

%% Creating a deformation matrix
% defining matrix T
T = [1 0 10
    0 1 20
    0 0 1];
disp(T)

% defining size of the image
num_pix_x = size(source_img,1);
num_pix_y = size(source_img,2);

% defining deformation field
def_field = defFieldFromAffineMatrix(T,num_pix_x,num_pix_y);

% resample using deformation field
img_resampled = resampImageWithDefField(source_img,def_field);

% display the transformed image
figure();
dispImage(img_resampled)

% check the value of a specific pixel
disp(img_resampled(255,255));

%% Resample using nearest neighbour and splinef2d

% nearest neighbour
img_nearest = resampImageWithDefField(source_img, def_field, 'nearest');
figure();
dispImage(img_nearest)

% splinef2d
img_spline = resampImageWithDefField(source_img, def_field, 'splinef2d');
figure();
dispImage(img_spline)

%% Difference images

% linear vs nearest
diff_img_ln = abs(img_nearest - img_resampled);
figure();
dispImage(diff_img_ln)

% linear vs spline
diff_img_ls = abs(img_spline - img_resampled);
figure();
dispImage(diff_img_ls)
disp(diff_img_ls(20,20))

%% Rotation about a point matrix

% close any open figures
close('all')

% define rotation matrix for rotation of 5 degrees around [10,10]
theta = 5;
p_coords = [10,10];
R = affineMatrixForRotationAboutPoint(theta, p_coords);
def_field = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y);
img_rotated = resampImageWithDefField(source_img,def_field);
figure();
dispImage(img_rotated)

% display image with source image intensity limits
smallest = min(source_img(:));
biggest = max(source_img(:));
int_lims = [smallest,biggest];
dispImage(img_rotated,int_lims)

%% 360 rotation transformation

% create for loop
for n = 1:71
    R = affineMatrixForRotationAboutPoint((n*5), [0,0]);
    rotation_transform = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y);
    rotation_image = resampImageWithDefField(img_rotated, rotation_transform);
    pause(0.05)
end

% show final image
dispImage(rotation_image)

%% Create rotation animation

% recreate affine matrix for 5 degrees around centre
coords = [(num_pix_x - 1)/2, (num_pix_y - 1)/2];
R = affineMatrixForRotationAboutPoint(5,coords);

% create deformation field
def_field = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y);
img_resampled = resampImageWithDefField(source_img, def_field);
dispImage(img_resampled);

% create current matrix as copy of rotation matrix
R_current = R;

% make a for loop
for n = 1:71
    
    % compose current matrix and rotation matrix
    R = R + R_current;
    
    % create deformation fields from n
    def_field = defFieldFromAffineMatrix(R,num_pix_x,num_pix_y);
    img_loop = resampImageWithDefField(source_img,def_field);
    dispImage(img_loop)
    
    pause(0.1)
    
end
