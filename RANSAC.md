# RANSAC for dynamic instability quantification

## Plugin steps

This plugin is designed to analyze 2D kymographs that can be in a directory and can have differnt image size. 
1) Open a directory containing the kymograph tif files and when prompted by the Napari viewer select the Mtrack reader function that opens it as a fake timelapse image containing different kymographs at different timesteps. 
2) Select the pre-trained segmentation model to segment the kymograph edges.
3) Select the parameters: pixel size in X and T (to compute the instability parameters in real units), number of tiles (see Segmentation of Kymographs), Max error (see Function Fits).
3) Select RANSAC model type (with minimum 2 points) or a quadratic model (with minimum 3 points).
4) Now hit run and it will segment the kymographs and perform the function fits.
5) To correct teh segmentation mistakes use the labels layer and hit 'Recompute current file fits'
6) If the results do not improve, you can do manual fits using Napari shapes layer and hit 'Recompute with manual functions', it will override the automatic fits.
7) The plot and table tabs contain the growth, shrink rates, catastrophe and rescue frequency as plot and table.

## Segmentation of Kymographs
We have a pre-trained U-Net model for segmenting kymographs. The U-Net model is applied on each timepoint of the kymograph, n_tiles is to enable fitting the each 2D image into memory and the default (1,1,1) shoudl work well if not you can set it to be (1,2,2) or even higher numbers to tile the image in the XY dimension. The result of this step is a labels layer that segments the edges of the kymograph.

## Function Fits on Segmented pixels
We have implemented RANSAC based function fits in our library, caped-ai-mtrack, using that library we can find multiple instances of linear functions in the image recursively. 
RANSAC is a non-deterministic algorithm that fits a function to data points by maximizing the number of data points that support the function fit (inliers).
If linear model is selected then we use linear function to find all the inliers, remove them from the fitting process and continue till all the growth/shrink events have been found in the image. If quadratic model is selected then we use quadratic function to find all the inliers and on the found inliers we fit linear function and repeat the process to find all the growth/shrink events. This method is based on our publication: doi:10.1038/s41598-018-37767-1

Max Error (in pixels) : This determines the maximum tolerance of RANSAC fits on both events, growth and shrinkage  events. A high maximum error value will classify more data points as inliers, a low maximum error will classify more data points as outliers.
Minimum number of time points : This determines the minimum number of data points that define a growth event. If you choose 20, only events that last for 20 time points will be considered to be a growth event. Shrinkage events, on the opposite, are much shorter and are allowed to last only 2 data points.
