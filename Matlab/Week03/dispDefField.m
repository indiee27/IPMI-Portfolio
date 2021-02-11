function dispDefField(def_field, spacing, plot_type)
%function to display a deformation field
%
%INPUTS:    def_field: the deformation field as a 3D array
%           spacing: the spacing of the grids/arrows in pixels [5]
%           plot_type: the type of plot to use, 'grid' or 'arrows' ['grid']

%set default values if parameters not set
if ~exist('spacing','var') || isempty(spacing)
    spacing = 5;
end
if ~exist('plot_type','var') || isempty(plot_type)
    plot_type = 'grid';
end

%calculate indices for plotting grid-lines/arrows
x_inds = 1:spacing:size(def_field,1);
y_inds = 1:spacing:size(def_field,2);

%check type of plot
switch plot_type
    case 'grid'
        
        %plot vertical grid lines
        plot(def_field(x_inds, :, 1)', def_field(x_inds, :, 2)', 'k');
        hold on
        %plot horizontal grid lines
        plot(def_field(:, y_inds, 1), def_field(:, y_inds, 2), 'k');
        hold off
        
    case 'arrows'
        
        %calculate grids of coords for plotting
        [Xs, Ys] = ndgrid(x_inds - 1, y_inds - 1);
        
        %calculate displacement field from deformation field
        disp_field_x = def_field(x_inds, y_inds, 1) - Xs;
        disp_field_y = def_field(x_inds, y_inds, 2) - Ys;
        
        %plot displacements using quiver
        quiver(Xs, Ys, disp_field_x, disp_field_y, 0);
        
    otherwise
        error('type must be grid or arrows');
end

%set axis limits and appearance
axis equal
axis tight