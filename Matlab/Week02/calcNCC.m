function NCC = calcNCC(A, B)
%function to calculate the normalised cross correlation between
%two images
%
%INPUTS:    A: an image stored as a 2D matrix
%           B: an image stored as a 2D matrix. B must the the same size as
%               A
%
%OUTPUTS:   NCC: the value of the normalised cross correlation
%
%NOTE: if either of the images contain NaN values these pixels should be
%ignored when calculating the NCC.

%use nanmean and nanstd functions to calculate mean and std dev of each
%image
mu_A = nanmean(A,'all');
mu_B = nanmean(B,'all');
sig_A = nanstd(A,1,'all'); %set flag to 1 as we want the population std dev
sig_B = nanstd(B,1,'all');

%calculate NCC using nansum to ignore nan values when summing over pixels
NCC = nansum((A - mu_A).*(B - mu_B),'all') / (sum(~isnan(A(:))) * sig_A * sig_B);