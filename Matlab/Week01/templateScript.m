clear
close all

%load lung MRI image
img = imread('lung_MRI_slice.png');

%check the data type of the image
disp(class(img));

%convert data type to double to avoid errors when processing integers
img = double(img);
disp(class(img));

% ***************
% ADD CODE HERE TO REORIENTATE THE IMAGE INTO 'STANDARD ORIENTATION'
% ***************

% display the image using the dispImage function 
dispImage(img)

%%

% ***************
% EDIT THE LINES BELOW TO CREATE A MATRIX REPRESENTING A
% TRANSLATION BY 10 PIXELS IN X AND 20 PIXELS IN Y
T = [1 0 0
    0 1 0
    0 0 1];
% ***************
disp(T)

% ***************
% ADD/EDIT CODE HERE TO:
%
% CREATE A DEFORMATION FIELD FROM THE AFFINE MATRIX
%(YOU NEED TO ADD THE REQUIRED INPUTS TO THE FUNCTION CALL)

def_field = defFieldFromAffineMatrix();

% RESAMPLE THE IMAGE USING THE DEFORMATION FIELD
img_resampled = resampImageWithDefField();
% ***************

% display the transformed image
dispImage(img_resampled)

% check the value of pixel 255,255 in the resampled image
disp(img_resampled(255,255));

%%

% ***************
% ADD CODE HERE TO:
%
% RESAMPLE THE IMAGE USING NEAREST NEIGHBOUR AND SPLINEF2D INTERPOLATION


% DISPLAY THE RESULTS IN SEPARATE FIGURES
figure();

figure();

% DISPLAY DIFFERENCE IMAGES BETWEEN THE NEW RESULTS AND THE ORIGINAL RESULT THAT USED
% LINEAR INTERPOLATION
figure();

figure();

% ***************

%%

% ***************
% ADD CODE HERE TO REPEAT THE STEPS ABOVE USING A TRANSLATION BY 10.5 PIXELS
% IN X AND 20.5 PIXELS IN Y
% ***************


%%

% ***************
% ADD CODE HERE TO REDISPLAY THE DIFFERENCE IMAGES WITH INTENSITY LIMITS OF [-20, 20]
% ***************

%%

% close any open figures
close('all')

% ***************
% ADD CODE HERE TO:
%
% USE THE ABOVE FUNCTION TO CALCULATE THE AFFINE MATRIX REPRESENTING AN ANTICLOCKWISE
% ROTATION OF 5 DEGREES ABOUT THE CENTRE OF THE IMAGE
R = affineMatrixForRotationAboutPoint()

% USE THIS ROTATION TO TRANSFORM THE ORIGINAL IMAGE

% DISPLAY THE RESULT USIGN THE INTENSITY LIMITS FROM THE ORIGINAL IMAGE
plt.figure()
int_lims_img = 
dispImage()
% ***************  

%%

% ***************
% ADD/EDIT CODE HERE TO APPLY THE SAME TRANSFORMATION AGAIN TO THE RESAMPLED IMAGE
% AND DISPLAY THE RESULT. REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE
% A FULL 360 DEGREES
for n = 1:71
    
    
    %add a short pause so the figure display updates
    pause(0.05);
end

% *************** 

% ***************   
% EDIT THE CODE ABOVE SO THAT IT USES A PADDING VALUE OF 0 INSTEAD OF NAN
% ***************
  
%%
  
% ***************
% ADD/EDIT CODE HERE TO REPEAT THE ABOVE ANIMATION BUT USING NEAREST NEIGHBOUR
% INTERPOLATION - FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY
% THE RESULT


%add a short pause after displaying the image so the figure display updates
pause(0.05);

% THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
% REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n = 1:71
    
    
    %add a short pause so the figure display updates
    pause(0.05);
end
% ***************

%%
  
% ***************
% ADD/EDIT CODE HERE TO REPEAT THE ABOVE ANIMATION BUT USING CUBIC
% INTERPOLATION - FIRST APPLY THE TRANSFORMATION TO THE ORIGINAL IMAGE AND DISPLAY
% THE RESULT


%add a short pause after displaying the image so the figure display updates
pause(0.05);

% THEN APPLY THE TRANSFORMATION TO THE RESAMPLED IMAGE AND DISPLAY THE RESULT
% REPEAT THIS 71 TIMES SO THAT THE IMAGE APPEARS TO ROTATE A FULL 360 DEGREES
for n = 1:71
    
    
    %add a short pause so the figure display updates
    pause(0.05);
end
% ***************

%%

% ***************
% ADD/EDIT CODE BELOW TO CREATE ANIMATIONS OF THE ROTATING IMAGES AS ABOVE, BUT COMPOSES THE
% TRANSFORMATIONS AT EACH STEP AND APPLIES THE COMPOSED TRANSFORMATION TO THE ORIGINAL IMAGE
%

% recreate the affine matrix corresponding to a rotation of 5 degrees about the centre of the image
R = affineMatrixForRotationAboutPoint(5, [(num_pix_x - 1)/2, (num_pix_y - 1)/2]);
  
% create deformation field, resample original image, and display result
def_field = defFieldFromAffineMatrix(R, num_pix_x, num_pix_y);
img_resampled = resampImageWithDefField(img, def_field);
dispImage(img_resampled, int_lims_img);
pause(0.05);

% create current matrix as copy of rotation matrix
R_current = R;

for n = 1:71
    
    % COMPOSE CURRENT MATRIX AND ROTATION MATRIX
    R_current =
    
    % CREATE DEFORMATION FIELD FOR CURRENT MATRIX, RESAMPLE ORIGINAL IMAGE, AND DISPLAY RESULT
    
    
    
    
    %add a short pause after displaying the image so the figure display updates
    pause(0.05)
end

% EDIT THE CODE ABOVE TO USE NEAREST NEIGHBOUR AND CUBIC INTERPOLATION INSTEAD OF LINEAR
% ***************

%%
% ***************
% COPY YOUR CODE ABOVE AND MODIFY IT TO USE PUSH INTERPOLATION INSTEAD OF PULL INTERPOLATION
% ***************