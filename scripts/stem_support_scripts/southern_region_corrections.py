import os
import sys
import numpy as np 
from osgeo import gdal,gdalconst
import pandas as pd 

def change_nodata_values(input_raster):
	"""Function to convert input nodata value to zeros."""
	#import gdal, gdalconst, numpy
	maskfile = gdal.Open(input_raster, gdalconst.GA_Update)
	maskraster = maskfile.ReadAsArray()
	maskraster = np.where((maskraster >= 0), maskraster, 0 ) 
	maskband = maskfile.GetRasterBand(1)
	maskband.WriteArray( maskraster )
	maskband.FlushCache()


change_nodata_values("/vol/v3/ben_ak/raster_files/southern_region_corrections/all_probability_layers_combined_nlcd_extent.tif")