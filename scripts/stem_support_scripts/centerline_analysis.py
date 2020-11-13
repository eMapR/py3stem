import os
import sys
import json
#import whitebox
from osgeo import ogr, gdal, osr
import numpy as np
#wbt = whitebox.WhiteboxTools()
from pysheds.grid import Grid
import geopandas as gpd 
from shapely.geometry import mapping
import matplotlib.pyplot as plt

class HydroTools(): 
	def __init__(self,input_dem,input_streams_layer,output_directory): 
		self.input_dem=input_dem
		self.input_streams_layer=input_streams_layer
		self.output_directory=output_directory

	def calc_elev_above_stream_euc(self): 
		output_filename = self.output_directory+'dist_above_stream_test.tif'
		print('working...')
		wbt.elevation_above_stream_euclidean(
		    self.input_dem, 
		    self.input_streams_layer, 
		    output_filename 
		    #callback='passed'
			)
		return None

def clean_center_line_raster(input_raster,output_directory):  
	# with rasterio.open(raster) as src: 
	# 	#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
	# 	transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
	# 	arr = src.read(1)#.astype('float')
	# 	arr[np.where(arr!=1)]=0
	outFileName=output_directory+'centerlines_no_data_corrected.tif'
	file =input_raster
	ds = gdal.Open(file)
	band = ds.GetRasterBand(1)
	arr = band.ReadAsArray()
	[cols, rows] = arr.shape
	arr_min = arr.min()
	arr_max = arr.max()
	arr_mean = int(arr.mean())
	arr_out = np.where((arr != 1), 0, arr)
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(outFileName, rows, cols, 1, gdal.GDT_UInt16)
	outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
	outdata.SetProjection(ds.GetProjection())##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(arr_out)
	outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
	outdata.FlushCache() ##saves to disk!!
	outdata = None
	band=None
	ds=None
class CreatePourPoints(): 
	def __init__(self,input_shapefile,output_directory): 
		self.input_shapefile=input_shapefile
		self.output_directory=output_directory

	def get_end_vertices(self): #should take as input a line shapefile
		full_shape=gpd.read_file(self.input_shapefile)
		print('working...')
		print(full_shape.head())
		pour_points = {}
		#note that the list of coordinates that make up the line go from the highest point (position 0) to the lowest point (position -1)
		for x in range(full_shape.shape[0]):  #get the number of rows which is the number of lines in the input
			objectid = full_shape.OBJECTID.iloc[x]
			low_point=mapping(full_shape.geometry.iloc[x])['coordinates'][-1] #low point
			pour_points.update({objectid:low_point})
		#high_point=mapping(full_shape.geometry.iloc[x])['coordinates'][0] #high point
		#print(low_point)
		#print(high_point)
		#print(pour_points)
		print('completed getting pour points')
		return pour_points
#delineate watersheds
class MakeWatersheds(): 
	def __init__(self,input_dem,output_directory): 
		self.input_dem=input_dem
		self.output_directory=output_directory

	def make_watersheds(self,point): 
		#>>> grid = Grid.from_raster('../data/dem.tif', data_name='dem')

		grid = Grid.from_raster(self.input_dem, data_name='dem')
		print(grid.crs)
		#grid.read_raster(os.path.split(self.input_dem)[0], data_name='dir')
		# Read raw DEM
		#grid = Grid.from_raster('../data/roi_10m', data_name='dem')

		# Fill depressions
		grid.fill_depressions(data='dem', out_name='flooded_dem')

		# Resolve flats
		grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
		# Determine D8 flow directions from DEM
		# ----------------------
		# Resolve flats in DEM
		#grid.resolve_flats('dem', out_name='inflated_dem')
		    
		# Specify directional mapping

		dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
		    
		# Compute flow directions
		# -------------------------------------
		grid.flowdir(data='dem', out_name='dir')#, dirmap=dirmap)
		#return grid,output
		#grid.view('dir')
		#plt.imshow(grid.view('dir'))

		#def make_watersheds(self,point): 
		# Specify pour point
		x, y = point
		print('x is: ',x)
		print('y is: ',y)
		#input_flow_dir=self.get_dem_and_infill()[1]
		# Delineate the catchment
		grid.catchment(data='dir', x=x, y=y, out_name='catch',
	                   recursionlimit=10000, xytype='label',dirmap=dirmap)
		demView = grid.view('catch', nodata=np.nan)

		#grid.to_raster(demView, self.output_directory+'dem_watershed_pysheds_catchment.tif')
		# Plot the result
		#grid.clip_to('catch')
		plt.imshow(grid.view('catch'))
		plt.show()
		plt.close()

def main():
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		input_dem = variables["input_dem"]
		input_streams_layer = variables["input_streams_layer"]
		output_directory = variables['output_directory']
		input_shapefile=variables['input_shapefile']
		pour_points=CreatePourPoints(input_shapefile,output_directory).get_end_vertices()
		print(pour_points[25021])
		#dem=MakeWatersheds(input_dem,output_directory).get_dem_and_infill()
		print('success')
		MakeWatersheds(input_dem,output_directory).make_watersheds(pour_points[25021])
		#HydroTools(input_dem,input_streams_layer,output_directory).calc_elev_above_stream_euc()
		#clean_center_line_raster(input_streams_layer,output_directory)
if __name__ == '__main__':
    main()