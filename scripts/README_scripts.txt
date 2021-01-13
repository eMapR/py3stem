
error_analysis.py
Required inputs- error_analysis.txt (param_file)
Env- this runs in a python 3 environment 

Functions: 

calc_confusion_matrix- 
This function relies on a few other functions and gets stats at predetermined points that were used for error analysis and then generates a confusion matrix and some error stats. 

calc_zonal_stats-
currently wrapped into nested for loops in the main() function, this thing calculates zonal stats but the nested for loops call a couple of other functions inside the GlacierSummary class which do some array processing

generate_random_pts-
This was used to generate the random points for error analysis on the glaciers. It requires a few rasters and does the generating by creating a masked binary raster then reading the raster into an array with xarray which includes the coordinates as x and y values. These are then randomly sampled from a list. 
