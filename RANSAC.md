# Plugin usage

## Tutorial
- A detailed tutorial can be found at [Demo](https://www.youtube.com/watch?v=MLLkC4Ls220&t=316s)


## Plugin Steps

- Analyze 2D kymographs from a directory, even with varying image sizes.
- Open a directory containing the kymograph TIFF files and select the Mtrack reader function in the Napari viewer to open them as a fake timelapse image.
- Choose a pre-trained segmentation model to segment kymograph edges.
- Configure parameters including pixel size in X and T, number of tiles, and maximum error.
- Select either a RANSAC model type (with a minimum of 2 points) or a quadratic model (with a minimum of 3 points).
- Click "Run" to segment kymographs and perform function fits.
- Correct segmentation mistakes using the labels layer and click "Recompute current file fits."
- If results don't improve, manually fit using the Napari shapes layer and click "Recompute with manual functions."
- View growth rates, shrink rates, catastrophe, and rescue frequency in the "Plot" and "Table" tabs.

## Outputs

- The Tabulour data contains dynamic instability measurments and can be saved with a right click as a csv.
- The plots are interactive and can be saved as well showing measurments for the whole experiment.


To display the movies below right click to display the controls.
### Open Data
To give the plugin a quick try get the data from the Napari Sample Data menu.
![Open Data](images/open_data_mod.mp4)

### First Run
For your first run use the pre-trained U-Net model to detect the kymograph edges and using RANSAC to fit functions with a max error of 0.01, Linear function and minimum number of points to be 2. 
![First Run](images/first_run_mod.mp4)

### Correction
For some kymographs the fits may not be adequate, in this case either change the parameters and try with different parameters and click on the Green button and it will change the results as RANSAC is a non-deterministic algorithm. If that does not work use the Napari shapes layer to remove the wrong fits and make your own fits and then use the Orange button to recompute using the manual fits and that will update the plots and the table.

Most importantly since you want your results to be in micrometer per second so do not forget to change the pixel size in X and T in the plugin menu as by default it is 1.
![Corrections](images/correction_mod.mp4)



## Segmentation of Kymographs

- Utilize a pre-trained U-Net model for kymograph segmentation.
- Apply the model to each timepoint of the kymograph.
- Adjust the number of tiles (e.g., (1,1,1) for default) to fit 2D images into memory.
- Generate a labels layer to segment kymograph edges.

## Function Fits on Segmented Pixels

- Implement RANSAC-based function fits in the caped-ai-mtrack library.
- RANSAC is a non-deterministic algorithm for fitting functions to data points.
- For the linear model, find inliers using a linear function, remove them from the fitting process, and continue until all growth/shrink events are found.
- For the quadratic model, find inliers with a quadratic function, fit a linear function to the inliers, and repeat to find all events.
- Control the "Max Error" (in pixels) to determine tolerance in RANSAC fits for growth and shrinkage events.
- Set the "Minimum Number of Time Points" to define growth events, e.g., choosing 20 means only events lasting 20 time points are considered growth events. Shrinkage events can be shorter, lasting only 2 data points.
