function [warped_image, def_field] = demonsReg(source, target, varargin)
%function to peform a registration between two 2D images using the demons
%algorithm
%
% USAGE:
%   demonsReg(source, target)
%   demonsReg(source, target, sigma_elastic, sigma_fluid)
%   demonsReg(source, target, sigma_elastic, sigma_fluid, num_lev)
%   demonsReg(source, target, sigma_elastic, sigma_fluid, num_lev, use_composition)
%   demonsReg(source, target, ..., Name, Value)
%   warped_image = demonsReg(...)
%   [warped_image, def_field] = demonsReg(..)
%
% DESCRIPTION:
%   Perform a registration between the 2D source image and the 2D target
% image using the demons algorithm. The source image is warped (resampled)
% into the space of the target image.
%   The values of sigma_elastic and sigma_fluid can optionally be provided
% to specify the amount of elastic and fluid regularistion to apply. these
% values specify the standard deviation of the Gaussian used to smooth the
% update (fluid) or displacement field (elastic). a value of 0 means no
% smoothing is applied. default values are:
% sigma_elastic = 1
% sigma_fluid = 1
%   The registration uses a multi-resolution scheme. The num_lev parameter
% can be used to specify the number of resolution levels to use. The
% default number of levels to use is 3.
%   The demons registration can be performed by either adding (classical
% demons) or composing (diffeomorphic demons) the updates at each
% iteration. Set the use_composition parameter to true to compose the
% updates at each iteration. The default value for use_composition is
% false, i.e. the updates will be added by default.
%   The final warped image and deformation field can be returned as outputs
% from the function.
%   There are a number of other parameters affecting the registration or
% how the results are displayed, which are explained below. These can be
% speficied using name-value pair arguments, e.g.:
% demonsReg(..., 'max_it', 500)
% The default values are given after the parameter name
%   use_target_grad = false
%       logical (true/false) value indicating whether the target image
%       gradient or source image gradient is used when calculating the
%       demons forces.
%   max_it = 1000
%       the maximum number of iterations to perform.
%   check_MSD = true
%       logical value indicating if the Mean Squared Difference (MSD)
%       should be checked for improvement at each iteration. If true, the
%       MSD will be evaluated at each iteration, and if there is no
%       improvement since the previous iteration the registration will move
%       to the next resolution level or finish if it is on the final level.
%   disp_freq = 5
%       the frequency with which to update the displayed images. the images
%       will be updated every disp_freq iterations. If disp_freq is set to
%       0 the images will not be updated during the registration
%   disp_spacing = 2
%       the spacing between the grid lines or arrows when displaying the
%       deformation field and update.
%   scale_update_for_display = 10
%       the factor used to scale the update field for displaying
%   disp_method_df = 'grid'
%       the display method for the deformation field.
%       can be 'grid' or 'arrows'
%   disp_method_up = 'arrows'
%       the display method for the update. can be 'grid' or 'arrows'


%parse inputs
%note - no checking on inputs is performed!
in_par = inputParser;

%set default values
default_sigma_elastic = 1;
default_sigma_fluid = 1;
default_num_lev = 3;
default_use_composition = false;
default_use_target_grad = false;
default_max_it = 1000;
default_check_MSD = true;
default_disp_freq = 5;
default_disp_spacing = 2;
default_scale_update_for_display = 10;
default_disp_method_df = 'grid';
default_disp_method_up = 'arrows';

%add parameters to parser
addRequired(in_par, 'source');
addRequired(in_par, 'target');
addOptional(in_par, 'sigma_elastic', default_sigma_elastic);
addOptional(in_par, 'sigma_fluid', default_sigma_fluid);
addOptional(in_par, 'num_lev', default_num_lev);
addOptional(in_par, 'use_composition', default_use_composition);
addParameter(in_par, 'use_target_grad', default_use_target_grad);
addParameter(in_par, 'max_it', default_max_it);
addParameter(in_par, 'check_MSD', default_check_MSD);
addParameter(in_par, 'disp_freq', default_disp_freq);
addParameter(in_par, 'disp_spacing', default_disp_spacing);
addParameter(in_par, 'scale_update_for_display', default_scale_update_for_display);
addParameter(in_par, 'disp_method_df', default_disp_method_df);
addParameter(in_par, 'disp_method_up', default_disp_method_up);

%parse inputs
parse(in_par, source, target, varargin{:});


%perform registration

%make copies of full resolution images
source_full = in_par.Results.source;
target_full = in_par.Results.target;

%loop over resolution levels
for lev = 1:in_par.Results.num_lev
    
    %resample images if needed
    if lev == in_par.Results.num_lev
        target = target_full;
        source = source_full;
    else
        resamp_factor = 2^(in_par.Results.num_lev - lev);
        target = imresize(target_full, 1/resamp_factor);
        source = imresize(source_full, 1/resamp_factor);
    end
    
    %if first level initialise def_field, disp_field
    if lev == 1
        [X, Y] = ndgrid(0:size(target,1)-1,0:size(target,2)-1);
        def_field(:,:,1) = X;
        def_field(:,:,2) = Y;
        disp_field_x = zeros(size(target));
        disp_field_y = zeros(size(target));
    else
        %otherwise upsample disp_field from previous level
        disp_field_x = 2 * imresize(disp_field_x, 2);
        disp_field_y = 2 * imresize(disp_field_y, 2);
        %recalculate def_field for this level from disp_field
        [X, Y] = ndgrid(0:size(target,1)-1,0:size(target,2)-1);
        def_field = [];%clear def field from previous level
        def_field(:,:,1) = X + disp_field_x;
        def_field(:,:,2) = Y + disp_field_y;
    end
    %initialise updates
    update_x = zeros(size(target));
    update_y = zeros(size(target));
    update_def_field = zeros(size(def_field));
    
    %calculate the transformed image at the start of this level
    warped_image = resampImageWithDefField(source, def_field);
    
    %store the current def field and MSD value to check for improvements at
    %end of iteration
    def_field_prev = def_field;
    prev_MSD = calcMSD(target, warped_image);
    
    %initialise 2D Gaussian filters for the elastic and fluid
    %regularisations
    elastic_filter = [];
    if in_par.Results.sigma_elastic > 0
        normal_dist = makedist('normal','sigma',in_par.Results.sigma_elastic);
        elastic_filter = pdf(normal_dist, floor(-3*in_par.Results.sigma_elastic):ceil(3*in_par.Results.sigma_elastic));
        elastic_filter = elastic_filter / sum(elastic_filter);
    end
    fluid_filter = [];
    if in_par.Results.sigma_fluid > 0
        normal_dist = makedist('normal','sigma',in_par.Results.sigma_fluid);
        fluid_filter = pdf(normal_dist, floor(-3*in_par.Results.sigma_fluid):ceil(3*in_par.Results.sigma_fluid));
        fluid_filter = fluid_filter / sum(fluid_filter);
    end
    
    %pre-calculate the image gradients. only one of source or target
    %gradients needs to be calculated, as indicated by use_target_grad
    if in_par.Results.use_target_grad
        [target_gradient_y, target_gradient_x] = gradient(target);
    else
        [source_gradient_y, source_gradient_x] = gradient(source);
    end
    
    %display current results:
    %figure 1 - source image (does not change during registration)
    %figure 2 - target image (does not change during registration)
    %figure 3 - source image transformed by current deformation field
    %figure 4 - deformation field
    %figure 5 - update
    figure(1)
    dispImage(source);
    figure(2)
    dispImage(target);
    figure(3);
    dispImage(warped_image);
    x_lims = xlim;
    y_lims = ylim;
    figure(4);
    dispDefField(def_field, in_par.Results.disp_spacing, in_par.Results.disp_method_df);
    xlim(x_lims);
    ylim(y_lims);
    figure(5);
    up_field_to_display = in_par.Results.scale_update_for_display * cat(3, update_x, update_y);
    up_field_to_display = up_field_to_display + cat(3, X, Y);
    dispDefField(up_field_to_display, in_par.Results.disp_spacing, in_par.Results.disp_method_up);
    xlim(x_lims);
    ylim(y_lims);
    
    %if first level pause so that user can position figure as desired
    if lev == 1
        input('position the figures as desired and then push enter to run the registration');
    end
    
    %main iterative loop - repeat until max number of iterations reached
    for it = 1:in_par.Results.max_it
        
        %calculate update from demons forces
        %
        %if using target image gradient use as is
        if in_par.Results.use_target_grad
            img_grad_x = target_gradient_x;
            img_grad_y = target_gradient_y;
        else
            %but if using source image gradient need to transform with
            %current deformation field
            img_grad_x = resampImageWithDefField(source_gradient_x, def_field);
            img_grad_y = resampImageWithDefField(source_gradient_y, def_field);
        end
        %calculate difference image
        diff = target - warped_image;
        %calculate denominator of demons forces
        denom = img_grad_x.^2 + img_grad_y.^2 + diff.^2;
        %calculate x and y components of numerator of demons forces
        numer_x = diff .* img_grad_x;
        numer_y = diff .* img_grad_y;
        %calculate the x and y components of the update
        update_x = numer_x ./ denom;
        
        update_y = numer_y ./ denom;
        %set nan values to 0
        update_x(isnan(update_x)) = 0;
        update_y(isnan(update_y)) = 0;
        
        
        %if fluid like regularisation used smooth the update
        if ~isempty(fluid_filter)
            update_x = conv2(fluid_filter', fluid_filter, padarray(update_x, ceil(3*in_par.Results.sigma_fluid)*[1 1], 'replicate'), 'valid');
            update_y = conv2(fluid_filter', fluid_filter, padarray(update_y, ceil(3*in_par.Results.sigma_fluid)*[1 1], 'replicate'), 'valid');
        end
        
        
        %update displacement field using addition (original demons) or
        %composition (diffeomorphic demons)
        if in_par.Results.use_composition
            %compose update with current transformation - this is done by
            %transforming (resampling) the current transformation using the
            %update. we can use the same function as used for resampling
            %images, and treat each component of the current deformation
            %field as an image
            %the update is a displacement field, but to resample an image
            %we need a deformation field, so need to calculate deformation
            %field corresponding to update.
            update_def_field(:,:,1) = update_x + X;
            update_def_field(:,:,2) = update_y + Y;
            %use this to resample the current deformation field, storing
            %the result in the same variable, i.e. we overwrite/update the
            %current deformation field with the composed transformation
            def_field(:,:,1) = resampImageWithDefField(def_field(:,:,1), update_def_field);
            def_field(:,:,2) = resampImageWithDefField(def_field(:,:,2), update_def_field);
            %update disp field from def field
            disp_field_x = def_field(:,:,1) - X;
            disp_field_y = def_field(:,:,2) - Y;
            %replace nans in disp field with 0s
            disp_field_x(isnan(disp_field_x)) = 0;
            disp_field_y(isnan(disp_field_y)) = 0;
        else
            %add the update to the current displacement field
            disp_field_x = disp_field_x + update_x;
            disp_field_y = disp_field_y + update_y;
        end
        
        
        %if elastic like regularisation used smooth the disp field
        if ~isempty(elastic_filter)
            disp_field_x = conv2(elastic_filter', elastic_filter, padarray(disp_field_x, ceil(3*in_par.Results.sigma_elastic)*[1 1], 'replicate'), 'valid');
            disp_field_y = conv2(elastic_filter', elastic_filter, padarray(disp_field_y, ceil(3*in_par.Results.sigma_elastic)*[1 1], 'replicate'), 'valid');
        end
        
        %update deformation field from disp field
        def_field(:,:,1) = disp_field_x + X;
        def_field(:,:,2) = disp_field_y + Y;
        
        %transform the image using the updated deformation field
        warped_image = resampImageWithDefField(source, def_field);
        
        %display current results (no need to update figures 1 and 2)
        if rem(it, in_par.Results.disp_freq) == 0
            figure(3);
            dispImage(warped_image);
            figure(4);
            dispDefField(def_field, in_par.Results.disp_spacing, in_par.Results.disp_method_df);
            xlim(x_lims);
            ylim(y_lims);
            figure(5)
            up_field_to_display = in_par.Results.scale_update_for_display * cat(3, update_x, update_y);
            up_field_to_display = up_field_to_display + cat(3, X, Y);
            dispDefField(up_field_to_display, in_par.Results.disp_spacing, in_par.Results.disp_method_up);
            xlim(x_lims);
            ylim(y_lims);
        end
        
        %calculate NMSD between target and warped image
        MSD =  calcMSD(target, warped_image);
        
        %display numerical results
        fprintf('Level %d, Iteration %d: MSD = %f\n', lev, it, MSD);
        
        %check for improvement in MSD if required
        if in_par.Results.check_MSD && MSD >= prev_MSD
            %restore previous results and finish level
            def_field = def_field_prev;
            warped_image = resampImageWithDefField(source, def_field);
            break;
        end
        
        %update previous values of def_field and MSD
        def_field_prev = def_field;
        prev_MSD = MSD;
        
    end
    
end

%display final result
figure(3);
dispImage(warped_image);
figure(4);
dispDefField(def_field, in_par.Results.disp_spacing, in_par.Results.disp_method_df);
xlim(x_lims);
ylim(y_lims);
figure(5)
up_field_to_display = in_par.Results.scale_update_for_display * cat(3, update_x, update_y);
up_field_to_display = up_field_to_display + cat(3, X, Y);
dispDefField(up_field_to_display, in_par.Results.disp_spacing, in_par.Results.disp_method_up);
xlim(x_lims);
ylim(y_lims);
