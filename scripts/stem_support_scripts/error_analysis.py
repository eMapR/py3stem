import os
import sys 
from osgeo import ogr 
import json 
from osgeo import gdal

import geopandas as gpd 
from rasterstats import zonal_stats,point_query
import rasterio
import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import pickle 
import matplotlib as mpl
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import xarray as xr
import dask
from datetime import datetime
import geopandas as gpd 
from random import seed
from random import randint
import random
from affine import Affine
from rasterio.features import shapes
from osgeo import ogr, gdal, osr
from osgeo.gdalnumeric import *  
from osgeo.gdalconst import * 
import fiona
from shapely.geometry import shape
import rasterio.features
import build_ref_dataset as ref_dataset
import re
import glob
import pyParz
import make_plots

def rasterize(shp,resolution,output_dest,extent_bounds): 
	"""Convert vector to raster."""
	input_shp = ogr.Open(shp)
	shp_layer = input_shp.GetLayer()
	extent_raster = rasterio.open(extent_bounds)
	
	pixel_size = resolution
	xmin, xmax, ymin, ymax = extent_raster.bounds#extent_raster.GetExtent()
	head,tail = os.path.split(shp)
	output_raster = output_dest+tail[:-4]+'.tif'
	print(output_raster)
	ds = gdal.Rasterize(output_raster, shp, xRes=pixel_size, yRes=pixel_size, 
	                    burnValues=1, outputBounds=[xmin, ymin, xmax, ymax], 
	                    outputType=gdal.GDT_Byte)
	ds = None
	return output_raster

def calc_zonal_stats(raster,shp,resolution,stat,source,transform_info,class_code): 
	"""Calculate pixel counts inside polygons."""
	geo_df = gpd.read_file(shp)
	try: 
		if raster.endswith('.tif'): 
			print('raster is',raster)
			with rasterio.open(raster,'r') as src: 
				#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
				transform = (src.bounds[0],float(resolution),0.0,src.bounds[3],0.0,-1*float(resolution))
				arr = src.read(1)#.astype('float')
				if ('stem' in source.lower()): 
					#print('we are binarizing the input raster')
					print('binarizing...')
					arr = np.where(arr==class_code,1,0)#input_arr[input_arr==12,1]
				elif (np.max(arr)>1):  
					print('binarizing...')
					arr=np.where((arr>class_code) & (arr<10),1,0)
				else:
					print('Assuming binary data input. Changing non-one values to zero')
					print(arr.min())
					print(arr.mean())
					print(arr.max()) 
					arr[arr!=1]=0
					arr[arr!=1.0]=0
		else: 

			arr = raster
			transform = transform_info
	except Exception as e: 
		print('Entered the except statement and the error that got us here was:')
		print(e)
		arr = raster
		transform = transform_info
	#rasterstats zonal stats produces a list of dicts, get the value
	stats = zonal_stats(geo_df,arr,stats=stat,transform=transform,nodata=255) #gets the values at the raster pixels for the given points
	output_geodf = geo_df.join(pd.DataFrame(stats))#.drop(['left','right','top','bottom'])
	#print('output_geodf stats')
	#print(output_geodf)
	#print(output_geodf.shape)
	#rename cols so they don't get angry when we join
	old_names = output_geodf.columns
	new_names = [source+'_'+i for i in old_names]
	column_names = dict(zip(old_names,new_names))
	output_geodf.rename(columns=column_names,inplace=True)


	#output_geodf = output_geodf.rename(columns={'count':source+'_count'},inplace=True)
	
	return output_geodf
	

def read_pickles(*argv): 
	#head,tail = os.path.split(raster_2)
	#actual_file = argv[9]+f'{argv[5]}_{argv[10]}_zonal_stats_df'
	predicted_file = argv[9]+f'{argv[7]}_{argv[6]}_{argv[10]}_zonal_stats_df'
	print(f'predicted file is {predicted_file}')
	if (argv[8].lower()=='true') and not (os.path.exists(predicted_file)): 
		print('pickling...')
		#generate zonal stats and pickle
		
		#if not os.path.exists(predicted_file): 
		#print(f'creating new files {actual_file} and {predicted_file}')
		#actual = calc_zonal_stats(argv[0],argv[2],argv[3],argv[4],argv[5])
		predicted = calc_zonal_stats(argv[1],argv[2],argv[3],argv[4],argv[6],argv[11],None) #added None 10/12/2020 because the calc_zonal_stats was modified for other uses
		#df = pd.concat([stem_df,rgi_df],axis=1)
		#actual_ds = pickle.dump(actual, open(actual_file, 'ab' ))
		predicted_ds = pickle.dump(predicted,open(predicted_file,'ab'))
		return predicted #remove actual 
		# else: 
		# 	print('both files already exist')
			
	else: #read in the pickled df if its the same data
		print('reading from pickle...')
		#actual_df = pickle.load(open(actual_file,'rb'))
		predicted_df = pickle.load(open(predicted_file,'rb'))
		#print(predicted_df.head())
		#print(predicted_df.columns)
		#print(actual_df.head())
		return predicted_df #removed the actual df 
#argv order: #0:nlcd_raster,1:stem_raster,2:random_pts,3:resolution,4:stat,5:actual_source,6:predicted_source,7:model_run,8:write_to_pickle,9:pickle_dir,10:modifier)

def calc_confusion_matrix(*argv):#actual_source,predicted_source,stat,): 
	"""Calculate a confusion matrix to compare nlcd or rgi and classification."""
	data = read_pickles(*argv)
	#print(data)
	actual = ref_dataset.get_csvs(argv[11],'discard','y').replace(np.nan,0)#previously getting values from raster, now getting from ref dataset csv

	predicted = data.replace(np.nan,0)#read_pickles(*argv)[1].replace(np.nan,0)#calc_zonal_stats(predicted_raster,shp,resolution,stat,predicted_source)
	predicted.index = np.arange(1,len(predicted)+1)
	predicted = predicted[predicted.index.isin(actual.index)] 
	
	#print('actual is: ',actual)
	#print('predicted is: ',predicted)
	actual_col = actual['binary']#actual[str(argv[5]+'_'+argv[4])]
	predicted_col = predicted[str(argv[6]+'_'+argv[4])]
	#print(actual_col.unique())
	#print(predicted_col.unique())
	actual_ls = [float(i) for i in list(actual_col)]
	predicted_ls = [float(i) for i in list(predicted_col)]
	actual_ids = actual.index
	predicted_ids = predicted.index
	

	#get the points that are not agreeing so we can see where in the world that is happening
	incorrect = pd.DataFrame([actual_ids,predicted_ids,list(actual.lat),list(actual.lon),actual_ls,predicted_ls]).T
	incorrect.columns=['actual_id','pred_id','lat','lon','actual_val','pred_val']
	print(incorrect.shape)
	incorrect=incorrect[incorrect['actual_val']!=incorrect['pred_val']]
	false_negatives = incorrect[incorrect['actual_val']>incorrect['pred_val']]
	false_positives = incorrect[incorrect['actual_val']<incorrect['pred_val']]
	print(false_positives)
	print(incorrect.shape)
	

	labels = sorted(list(set(list(actual_col)+list(predicted_col))))

	results = confusion_matrix(actual_ls, predicted_ls,labels) 
	print('#########################################')
	print(results)
	print(type(results))
	#disp = plot_confusion_matrix(None,actual_ls,predicted_ls,display_labels=labels,cmap=plt.cm.Blues)
	#fig,(ax,ax1) = plt.subplots(nrows=1,ncols=2)
	ax=plt.subplot()
	sns.heatmap(results,annot=True,ax=ax,fmt='g',vmin=0.0,vmax=400.0,cmap='Greys')
	#ax.collections[0].colorbar.ax.set_ylim(0,400)

	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	ax.set_title(f'{argv[6]} {argv[10]}') 
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	#print(results) 
	# print ('Accuracy Score :',accuracy_score(actual_ls, predicted_ls))
	# print ('Report : ')
	# print (classification_report(actual_ls, predicted_ls))
	print(classification_report(actual_ls,predicted_ls))
	classification_output = pd.DataFrame(classification_report(actual_ls,predicted_ls,output_dict=True)).transpose().reset_index().rename(columns={'index':'stat'})
	classification_output['model_accuracy'] = accuracy_score(actual_ls,predicted_ls)
	print('output is: ', classification_output)
	print(classification_output.index)
	#print('output is: ',type(classification_report(actual_ls,predicted_ls,output_dict=True)))
	# 	report = classification_report(y_test, y_pred, output_dict=True)
	# and then construct a Dataframe and transpose it:

	# df = pandas.DataFrame(report).transpose()
	plt.show()
	plt.close('all')
	return classification_output, incorrect, false_positives, false_negatives

def make_percent_change_map(stem_raster,rgi_raster,shp,resolution,output_dir,boundary,zoom,pickle_dir,read_from_pickle,stat): 
	"""A helper function for calc_zonal_stats."""
	head,tail = os.path.split(stem_raster)

	if read_from_pickle.lower()=='true': 
		print('pickling...')
		#generate zonal stats and pickle
		output_file = pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'
		if os.path.exists(output_file): 
			pass
		else: 
			stem_df = calc_zonal_stats(stem_raster,shp,resolution,stat,'stem')
			rgi_df = calc_zonal_stats(rgi_raster,shp,resolution,stat,'rgi')
			df = pd.concat([stem_df,rgi_df],axis=1)
			pickle_data=pickle.dump(df, open(output_file, 'ab' ))
	elif os.path.exists(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'): #read in the pickled df if its the same data
		print('reading from pickle...')
		df = pickle.load(open(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df','rb'))
	else: 
		print(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'+'does not exist, please make sure the settings in the param file are correct')
	#calculate the percent error aggregated by cell (pixelwise doesn't make sense because its binary)
	df['pct_err'] = (((df['stem_count']-df['rgi_count'])/df['rgi_count'])*100)
	#rename a col to geometry because the plot function wants that
	df.rename(columns={'rgi_geometry':'geometry'},inplace=True)
	#get rid of garbage 
	df = df.drop(['stem_left','stem_top','stem_right','stem_bottom','stem_geometry','rgi_left','rgi_top','rgi_right','rgi_bottom'],axis=1)
	#select a subset by getting rid of infs
	df_slice = df.replace([np.inf, -np.inf],np.nan).dropna(axis=0)#df.query('stem_count!=0')#[df['stem_count']!=0 and df['rgi_count']!=0]
	#read in plotting shapefiles
	inset = gpd.read_file(boundary)
	background = gpd.read_file(zoom)
	#do the plotting 
	fig,ax = plt.subplots()
	#make the colorbar the same size as the plot
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right",size="5%",pad=0.1)
	left, bottom, width, height = [0.1, 0.525, 0.25, 0.25]
	ax1 = fig.add_axes([left, bottom, width, height])
	#specify discrete color ramp 
	cmap = mpl.colors.ListedColormap(['#005a32','#238443','#41ab5d','#78c679','#addd8e','#d9f0a3','#ffffcc',#'#ffffcc','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32',
	'#F2B701','#E73F74','#180600','#E68310','#912500','#CF1C90','#f23f01',
	'#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']) 
	#'#855C75','#D9AF6B','#AF6458','#736F4C','#526A83','#625377','#68855C','#9C9C5E','#A06177','#8C785D','#467378','#7C7C7C'])
	#'#5F4690','#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666'])#'#5D69B1','#52BCA3','#99C945','#CC61B0','#24796C','#DAA51B','#2F8AC4','#764E9F','#ED645A','#CC3A8E','#63665b'])#["#f5b460","#F1951C","#a86813", "#793200", "#004039","#006B5F", "#62BAAC","#ba6270"])#'#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'])#
	norm = mpl.colors.BoundaryNorm([-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap.N) 
	background.plot(color='lightgray',ax=ax)
	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax,legend=True,cax=cax,cmap=cmap,norm=norm)
	ax.set_title(f'{tail[0:8]} model run AK southern processing region percent error')

	inset.plot(color='lightgray',ax=ax1)
	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax1,cmap=cmap,norm=norm)
	ax1.get_xaxis().set_visible(False)
	ax1.get_yaxis().set_visible(False)
	plt.tight_layout() 
	plt.show()
	plt.close('all')

def nlcd_disagree_summary(input_raster): 
	"""Summarize the nlcd classes where LT/stem and rgi disagree."""
	with rasterio.open(input_raster) as src: 
		arr = src.read(1)#.astype('float')
		arr[arr<0] = 0
		arr = arr.astype('int32')
		bins=np.unique(arr.flatten())[1:]
		bin_count = np.bincount(arr.flatten())
		bin_count = bin_count[bin_count != 0][1:]
		head,tail = os.path.split(input_raster)
		fig,ax = plt.subplots()
		ax.bar(bins,bin_count,width=1,align='center',color='red',edgecolor='black')#,tick_label=bins_str)
		ax.set_xticks(bins)
		ax.set_xlabel('NLCD Class')
		ax.set_ylabel('Counts')
		ax.set_title(tail[:-4])
		plt.show()

def reclassify(input_raster,nlcd_version,reclass_value): 
	with rasterio.open(input_raster) as src: 
		arr = src.read()
		profile = src.profile
		#print(profile)
		class_pairs = {0:0,11:7,12:9,22:18,23:16,31:27,41:36,42:38,43:46,51:48,52:56,71:67,72:78,90:87,95:92}
		#do the reclassify
		if nlcd_version.lower()=='old': 
			for k,v in class_pairs.items(): 
				arr[np.where(arr==k)]=v
		elif nlcd_version.lower()=='new': 
			inverted = {v:k for k,v in class_pairs.items()}
			for k,v in inverted.items(): 
				arr[np.where(arr==k)] = v
		elif nlcd_version.lower() =='binary': 
			#this is for binary rasters
			arr[np.where(arr==1)] = reclass_value
		#use if you want to assign your own dictionary with key being the original value and value being the remapped value
		elif nlcd_version.lower() == 'user': 
			try: 
				input_dict = dict(map(lambda x: x.split(':'), reclass_value.split(',')))  
				input_dict = {int(k):int(v) for k,v in input_dict.items()}

				print(input_dict)
			except ValueError: 
				try: 
					input_dict = dict(map(lambda x: x.split(':'), reclass_value.split(', '))) 
					input_dict = {int(k):int(v) for k,v in input_dict.items()}
				except ValueError: 
					print('It looks like your formatting is incorrect. Please check that values are added as key:value, or key:value, ')
			for k,v in input_dict.items(): 
				arr[np.where(arr==k)] = v 

	output_file = input_raster[:-4]+'_reclassify.tif'
	with rasterio.open(output_file, 'w', **profile) as dst: 
		dst.write(arr)
class GeneratePoints(): 
	"""A class to hold functions for cleaning a raster and then generating random points for stratified random sampling."""

	def __init__(self,input_raster,dif_raster,output_dir): 
		self.input_raster=input_raster
		self.dif_raster=dif_raster
		self.output_dir=output_dir
	def pad_raster(self): 
		"""Make a difference raster that is not filled with nodata values."""
		#first get the compilation of the three categories and pad it 
		
		with rasterio.open(self.dif_raster) as src2: 
			dif_arr = (xr.open_rasterio(src2,chunks=(1,(2917*5),(5761*5)))) 
			dif_arr_subset = (dif_arr.where(dif_arr>0,drop=True)).astype('uint8')
			#print(dif_arr_subset.shape)
		with rasterio.open(self.input_raster) as src1:#, open(self.dif_raster) as src2: 
			full_extent = xr.open_rasterio(src1,chunks=('auto'))#.shape
			print(full_extent.shape)
			output_arr = full_extent*dif_arr_subset
			profile = src1.profile

			#print(output_arr) 
			# output_file = self.input_raster[:-4]+'_glaciers_removed.tif'
			# with rasterio.open(output_file, 'w', **profile) as dst: 
			# 	dst.write(output_arr)
		return output_arr
	def get_class_size(self): 
		"""Get the valid pixel count (1's) from a binary raster."""
		with rasterio.open(self.input_raster) as src: 
			arr = src.read(1)#.astype('float')
			arr[arr<0] = 0
			print(np.sum(arr))
	def reduce_raster_size(self): 
		with rasterio.open(self.dif_raster) as src: 
			arr = src.read(1)
			print(arr.shape)
			arr = np.where(arr==0)
			print(arr)
			#arr = arr.flatten()
			print(arr.shape)
	def select_random_pts_from_raster(self,target_value,num_pts):
		"""From an input raster select a random subset of points (pixels)."""

		#read in the raster
		with rasterio.open(self.dif_raster) as src: 
			arr = xr.open_rasterio(src,chunks='auto')#1,(2917*6),(5761*6))) #was 1000000 1/31/2021
			profile = src.profile

			print(arr.shape)
			if len(arr.shape) > 2: 
				try: 
					arr = arr.squeeze(axis=0)
				except: 
					arr = arr[0, :, :]

			else: 
				print('the shape of your array is: ', arr.shape)
			arr= arr.where(arr==target_value,drop=True) #changed so that the user can select the value you want 
			new_arr = arr.stack(z=('x','y')).reset_index('z')
			new_arr = new_arr.dropna('z',how='any')
			subset_list = []
			coords_list = []
			seed(1)
			indices = np.random.choice(range(0, new_arr.shape[0]), num_pts,replace=False)
			for i in indices: 
				i = int(i)
				coords_list.append(tuple([float(new_arr[int(i)].coords['y'].values),float(new_arr[int(i)].coords['x'].values),float(new_arr[int(i)].coords['band'].values)]))#append a tuple of coords in the form row,col
			
			df = pd.DataFrame(coords_list,columns=['lat','lon','value'])#{'lat':y_vals.tolist(),'lon':x_vals.tolist()},index=range(1,51))
			gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat))
			gdf.crs='EPSG:3338'
			gdf['id'] = range(1,gdf.shape[0]+1)
			#write out 
			out_filename = os.path.split(self.dif_raster)[1][:-4]
			#write a shapefile
			gdf.to_file(self.output_dir+out_filename+f'_{num_pts}_class_{target_value}.shp')
			gdf.to_csv(out_filename+'.csv')

		return gdf
class Polygonize(): 
	def __init__(self,input_raster,output_dir): 
		self.input_raster = input_raster
		self.output_dir = output_dir
	def raster_to_polygon(self): 
		from rasterio.features import shapes
		mask = None
		with rasterio.Env():
		    with rasterio.open(self.input_raster) as src:
		        image = src.read(1) # first band
		        #print(image.shape)
		        # arr = image[image == 1]
		        # print(arr.shape)
		        #n1, n2 = np.arange(50), np.arange(50)

		        #b = image[n1[:,None], n2[None,:]]
		        results = (
		        {'properties': {'raster_val': v}, 'geometry': s}
		        for i, (s, v) 
		        in enumerate(
		            shapes(image, mask=mask, transform=src.transform)))

		# 		print shape(geoms[0]['geometry'])
		# POLYGON ((202086.577 90534.35044406779, 202086.577 90498.96206999999, 202121.9653740678 90498.96206999999, 202121.9653740678 90534.35044406779, 202086.577 90534.35044406779))
# Create geopandas Dataframe and enable easy to use functionalities of spatial join, plotting, save as geojson, ESRI shapefile etc.
		#shape(geoms[0]['geometry']
		from shapely.geometry import shape
		geoms = list(results)
		print(geoms[0])
		#geoms = [shape(i['geometry']) for i in list(results)]
		#geoms = list(results)
		#print(geoms[0])
		gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms,crs='EPSG:3338')
		print(gpd_polygonized_raster.head())
		print(gpd_polygonized_raster.shape)
		output_gdf = gpd_polygonized_raster.loc[gpd_polygonized_raster['raster_val'] == 1]
		output_gdf['geometry'] = output_gdf['geometry'].buffer(0.0)
		print(output_gdf.shape)
		output_gdf.to_file(self.output_dir+self.input_raster[:-4]+'.shp')
		#from https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons

class GlacierSummary(): 
	def __init__(self,input_raster,mask_raster,region):
		self.input_raster=input_raster
		self.mask_raster=mask_raster
		self.region = region
		
	def mask_rasters(self,class_code,resolution): 
		"""Get two rasters, binarize a LULC map raster for a class (class_code) and then select subset based on mask."""
		ds1 = gdal.Open(self.input_raster)
		input_arr = ds1.ReadAsArray()
		#print(input_arr)
		#print(f'input_arr max is {input_arr.max()} and should be 95')
		ulx, xres, xskew, uly, yskew, yres  = ds1.GetGeoTransform()

		transform = (ulx,resolution,0.0,uly,0.0,-1*resolution)
		#get just the glacier class as a binary raster
		input_arr[input_arr>class_code] = 1#np.where(input_arr==class_code,1,0)#input_arr[input_arr==12,1]
		bin_arr = input_arr
		bin_arr[bin_arr!=1] = 0 
		#input_arr[np.isnan(input_arr)] = 0 
		ds2 = gdal.Open(self.mask_raster)

		#ds3 = gdal.Open(self.region)
		mask_arr = ds2.ReadAsArray()
		#mask_arr[np.isnan(mask_arr)] = 0 
		if mask_arr.max() > 1: 
			mask_arr[mask_arr>1] = 1 
		mask_arr[mask_arr!=1] = 0
		#region_arr = ds3.ReadAsArray()
		#region_arr[region_arr!=1]=0
		try: 
			masked = bin_arr * mask_arr #* region_arr
			#print(f'masked max is {masked.max()}')
			print(masked.shape)
			#print(masked)
			#masked[masked<0]=0
			ds1=None
			ds2=None
			#ds3=None
			return masked,transform
		except ValueError: 
			print('fail')
			raise

	def mask_and_sum_rasters(self,arr,spatial_res): 
		"""Sum a binary array which came from a binary raster."""

		mask_sum = np.sum(arr)
		mask_area = mask_sum * spatial_res *spatial_res #change pixel count to area by multiplying 30m x 30m x pixel count
		mask_area = mask_area/1000000
		return mask_area
		

# def parallel_rasters(args): 
# 	classified_raster,ref_raster = args
# 	year = (os.path.split(classified_raster)[1]).split('_')[1]
# 	glacier_area = GlacierSummary(classified_raster,ref_raster).mask_rasters()
# 	return year,glacier_area
def run_in_parallel(args): 
	region,raster_dir,qualifier,mask_raster=args
	for file in glob.glob(raster_dir+qualifier):
		print('file is: ',file)
		region_mask=GlacierSummary(file,mask_raster,region).mask_rasters(12,30.0) 
		region_sum = GlacierSummary(None,None,None).mask_and_sum_rasters(region_mask,30.0)
		print(region_sum)
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		shapefile = variables["shapefile"]
		resolution = int(variables["resolution"])
		output_dir = variables["output_dir"]
		pickle_dir = variables["pickle_dir"]
		raster_dir = variables['raster_dir']
		ref_raster = variables["ref_raster"] #previously "/vol/v3/ben_ak/raster_files/glacier_velocity/rgi_cnd_tiles_southern_region_dissolve.tif",
		classified_raster = variables["classified_raster"]
		nlcd_raster = variables["nlcd_raster"]
		boundary = variables["boundary"]
		#zoom = variables["zoom"]
		#hist_raster = variables["hist_raster"]
		random_pts = variables["random_pts"]
		write_to_pickle = variables["write_to_pickle"]
		stat = variables["stat"]
		actual_source = variables["actual_source"]
		predicted_source = variables["predicted_source"]
		model_run = variables["model_run"]
		nlcd_version = variables["nlcd_version"]
		modifier = variables["modifier"]
		reclass_value = variables["reclass_value"]
		reclass_dict = variables["reclass_dict"]
		uncertainty_layer = variables["uncertainty_layer"]
	t0 = datetime.now()
	#generate random reference points
	pts = GeneratePoints(None,ref_raster,output_dir)
	output_coords = pts.select_random_pts_from_raster(1,50000) #the extra arg can be 0 or 1 currently with 0 being select anything that isn't glacier and 1 being glacier. The other number is the # of pts you want to generate

	#calculate regional total glacier area
	# year_dict = {}
	# for file in sorted(glob.glob(raster_dir+'*glacier_probabilities_no_low_class_vote.tif')): 
	# 	#if 'tcb' in file: #removed 12/15/2020 because working on probabilites not nlcd
	# 		print(file)
	# 		#year = os.path.split(file)[1][:4]
	# 		year = (os.path.split(file)[1]).split('_')[1]#.split("model_", 1)[1].split("_", 1)[0].strip()#re.search(r"model_ (\d{4})", os.path.split(file)[1]).group(1) #get the year
	# 		print(f'year is {year}')
	# 		glacier_arr = GlacierSummary(file,ref_raster,None).mask_rasters(3,resolution) #this just gets the total number of pixels for one year and one uncertainty level 
	# 		glacier_area = GlacierSummary(None,None,None).mask_and_sum_rasters(glacier_arr[0],resolution)
	# 		print(glacier_area)
	# 		year_dict.update({year:glacier_area})
	# 	#else: 
	# 	#	print('That file was for tcb, passing')
	# print(year_dict)
	# output_df = pd.DataFrame(year_dict,index=range(len(year_dict)))
	# output_df.to_csv(output_dir+'southern_region_glacier_probablities_model_no_low_class_all_years.csv')

	# print(year_dict)	
	# df = pd.DataFrame(year_dict,index=[0])
	# df.to_csv(output_dir+(os.path.split(ref_raster)[1][:-4])+'_clipped.csv')
	#generate areas for climate regions
	#climate_regions = glob.glob(boundary+'*.tif')
	#print(climate_regions)
	#region_areas = pyParz.foreach(climate_regions,run_in_parallel,args=[raster_dir,'*2016_full_run_vote.tif',ref_raster],numThreads=5)
	# df_dict = {}
	# arr_dict = {}
	# #make the arrays
	# df_dict = {}
	# for region_file in sorted(glob.glob(boundary+'*.tif')): 
	# 	#if ('denali' in shp_file) or ('wrangall' in shp_file) or ('glacier' in shp_file): 
	# 	uncertainty_level = make_plots.MakePlots(None,None,None,None,None).check_field(ref_raster)
	# 	output_file=output_dir+os.path.split(region_file)[1][:-4]+f'_{uncertainty_level}_2016_nlcd_climate_regions.csv'
	# 	if not os.path.exists(output_file): 
	# 		for file in sorted(glob.glob(raster_dir+'*2016_full_run_vote.tif')): 
	# 			year = (os.path.split(file)[1]).split('_')[1]
	# 			# masked = GlacierSummary(file,ref_raster).mask_rasters(12,resolution)
	# 			# arr = masked[0]
	# 			# transform = masked[1]
	# 			# stats_df=calc_zonal_stats(arr,shp_file,resolution,stat,f'stem_{year}',None,transform)
	# 			# print(stats_df)
	# 			region_mask=GlacierSummary(file,ref_raster,region_file).mask_rasters(12,30.0) 
	# 			region_sum = GlacierSummary(None,None,None).mask_and_sum_rasters(region_mask,30.0)
	# 			print(region_sum)
	# 			#sum_value = stats_df[f'stem_{year}_{stat}'][0]
	# 			#if sum_value == 0: 
	# 			#	break
	# 			#else: 
	# 			#	df_dict.update({year:sum_value})
	# 		print(df_dict)
	# 		df = pd.DataFrame(df_dict,index=[0])
	# 		df.to_csv(output_file)#output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_updated.csv')
	# 	else: 
	# 		print('that file exists')
	# for file in glob.glob(raster_dir+qualifier):
	# 	print('file is: ',file)
	# 	region_mask=GlacierSummary(file,mask_raster,region).mask_rasters(12,30.0) 
	# 	region_sum = GlacierSummary(None,None,None).mask_and_sum_rasters(region_mask,30.0)
	# 	print(region_sum)
	#generate glacier areas for national parks
	# df_dict = {}
	# arr_dict = {}
	# #make the arrays
	# df_dict = {}
	# for shp_file in sorted(glob.glob(boundary+'*.shp')): 
	# 	#if ('denali' in shp_file) or ('wrangall' in shp_file) or ('glacier' in shp_file): 
	# 	uncertainty_level = make_plots.MakePlots(None,None,None,None,None).check_field(ref_raster)
	# 	output_file=output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_2016_nlcd_climate_regions.csv'
	# 	if not os.path.exists(output_file): 
	# 		print(shp_file)
	# 		for file in sorted(glob.glob(raster_dir+'*2016_full_run_vote.tif')): 
	# 			print(file)
	# 			year = (os.path.split(file)[1]).split('_')[1]
	# 			masked = GlacierSummary(file,ref_raster).mask_rasters(12,resolution)
	# 			arr = masked[0]
	# 			transform = masked[1]
	# 			stats_df=calc_zonal_stats(arr,shp_file,resolution,stat,f'stem_{year}',None,transform)
	# 			print(stats_df)
	# 			sum_value = stats_df[f'stem_{year}_{stat}'][0]
	# 			if sum_value == 0: 
	# 				break
	# 			else: 
	# 				df_dict.update({year:sum_value})
	# 		print(df_dict)
	# 		df = pd.DataFrame(df_dict,index=[0])
	# 		df.to_csv(output_file)#output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_updated.csv')
	# 	else: 
	# 		print('that file exists')

	

	#reclassify(nlcd_raster,nlcd_version,reclass_dict)
	#nlcd_disagree_summary(stem_raster)
	#calc_zonal_stats(nlcd_raster,random_pts,resolution,stat,'nlcd')
	#error_arr = GeneratePoints(uncertainty_layer,None,None).get_class_size()
	######################################################################################################
	#make confusion matrix
	#error_stats=calc_confusion_matrix(None,classified_raster,random_pts,resolution,stat,actual_source,predicted_source,model_run,write_to_pickle,pickle_dir,modifier,uncertainty_layer)
	
	# #get a csv of the points that were incorrectly classified so we can see where they are located- this is the second output of the calc_confusion_matrix func
	# error_stats[1].to_csv(output_dir+modifier.replace(' ','_')+'_incorrect_points.csv')
	# #get csv of false positives
	# error_stats[2].to_csv(output_dir+modifier.replace(' ','_')+'_false_positives.csv')
	# error_stats[3].to_csv(output_dir+modifier.replace(' ','_')+'_false_negatives.csv')
	#####################################################################################################
	#extract_raster_pts(nlcd_raster,random_pts,resolution)
	# for file in glob.glob(boundary+'*.shp'): 
	# 	rasterize(file,resolution,output_dir,ref_raster)
	#####################################################################################################
	#calculate zonal stats
	# zonal_stats=calc_zonal_stats(classified_raster,shapefile,resolution,stat,'rgi',None,None)
	# print(zonal_stats)
	# area=(zonal_stats*resolution*resolution)/1000000
	#raster,shp,resolution,stat,source,transform_info,class_code):
	t1 = datetime.now()
	print((t1-t0)/60)
if __name__ == '__main__':
	main()
# {
# "shapefile": "/vol/v3/ben_ak/vector_files/stem_processing/rgi_cnd_tiles_southern_region_dissolve.shp",
# "resolution": "30",
# "output_dir":"/vol/v3/ben_ak/excel_files/error_analysis/", 
# "pickle_dir":"/vol/v3/ben_ak/param_files/script_params/",  
# "raster_dir":"/vol/v3/ben_ak/param_files_rgi/northern_region/output_files/",
# "ref_raster":"/vol/v3/ben_ak/raster_files/sp_data/northern_region_sp_06_thresh_55_times_high_certainty_final_extent.tif",
# "nlcd_raster":"/vol/v3/ben_ak/raster_files/glacier_velocity/rgi_cnd_tiles_southern_region_dissolve.tif", 
# "classified_raster":"/vol/v3/ben_ak/param_files_rgi/southern_region/output_files/models_2017_year_nlcd_original_2016_full_run_vote.tif", 
# "hist_raster":"/vol/v3/ben_ak/raster_files/glacier_velocity/2001_gte_5_reprojected_epsg_3338_30m.tif", 
# "boundary":"/vol/v3/ben_ak/raster_files/climate_region_rasters/", 
# "zoom":"/vol/v3/ben_ak/vector_files/ifsar_processing/rgi_cnd_tiles_northern_region_dissolve.shp", 
# "random_pts":"/vol/v3/ben_ak/vector_files/uncertainty_layers/2001_gte_10_rgi_nlcd_reclassify_class_three_only_AK_clipped.shp",
# "uncertainty_layer":"/vol/v3/ben_ak/py3stem/ref_data/2001_gte_10_rgi_nlcd_reclassify_class_three_only_AK_clipped.csv",
# "output_raster_dir":"/vol/v3/ben_ak/param_files_rgi/southern_region/certainty_levels_tifs/", 
# "write_to_pickle":"true", 
# "stat":"majority", 
# "actual_source":"stem", 
# "predicted_source":"2016-2017 STEM composite", 
# "model_run":"09262020", 
# "nlcd_version":"2016_NLCD_original_version", 
# "reclass_value":"1",
# "reclass_dict":"0:0,1:1,2:1,3:2,4:1,5:2,6:2,7:3",
# "modifier":"high certainty"
# }

# df_dict = {}
# 	for shp_file in glob.glob(boundary+'*.shp'): 
# 		for file in sorted(glob.glob(raster_dir+'*.tif')): 
# 			year = (os.path.split(file)[1]).split('_')[1]
# 			stats_df=calc_zonal_stats(file,shp_file,resolution,stat,f'stem_{year}',None)
# 			print(stats_df)
# 			df_dict.update({year:stats_df[f'stem_{year}_{stat}'][0]})
# 		print(df_dict)
# 		df = pd.DataFrame(df_dict,index=[0])
# 		df.to_csv(output_dir+os.path.split(shp_file)[1][:-4]+'.csv')

# 	#calculate regional total glacier area
# 	year_dict = {}
# 	for file in sorted(glob.glob(raster_dir+'*.tif')): 
# 		print(file)
# 		#year = os.path.split(file)[1][:4]
# 		year = (os.path.split(file)[1]).split('_')[1]#.split("model_", 1)[1].split("_", 1)[0].strip()#re.search(r"model_ (\d{4})", os.path.split(file)[1]).group(1) #get the year
# 		print(f'year is {year}')
# 		glacier_area = GlacierSummary(file,ref_raster).mask_rasters() #this just gets the total number of pixels for one year and one uncertainty level 
# 		print(glacier_area)
# 		year_dict.update({year:glacier_area})

# 	print(year_dict)	
# 	df = pd.DataFrame(year_dict,index=[0])
# 	df.to_csv(output_dir+(os.path.split(ref_raster)[1][:-4])+'.csv')

# #working as of 10/06/2020
# df_dict = {}
# 	for shp_file in sorted(glob.glob(boundary+'*.shp')): 
# 		#if ('denali' in shp_file) or ('wrangall' in shp_file) or ('glacier' in shp_file): 
# 		uncertainty_level = make_plots.MakePlots(None,None,None).check_field(ref_raster)
# 		output_file=output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_corrected.csv'
# 		if not os.path.exists(output_file): 
# 			print(shp_file)
# 			for file in sorted(glob.glob(raster_dir+'*full_run_vote.tif')): 
# 				print(file)
# 				year = (os.path.split(file)[1]).split('_')[1]
# 				masked = GlacierSummary(file,ref_raster).mask_rasters(12,resolution)
# 				arr = masked[0]
# 				transform = masked[1]
# 				stats_df=calc_zonal_stats(arr,shp_file,resolution,stat,f'stem_{year}',None,transform)
# 				print(stats_df)
# 				sum_value = stats_df[f'stem_{year}_{stat}'][0]
# 				# if sum_value == 0: 
# 				# 	break
# 				# else: 
# 				df_dict.update({year:sum_value})
# 			print(df_dict)
# 			df = pd.DataFrame(df_dict,index=[0])
# 			df.to_csv(output_file)#output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_updated.csv')
# 		else: 
# 			print('that file exists')

# for file in sorted(glob.glob(raster_dir+'*full_run_vote.tif')): 
# 		print(file)
# 		year = (os.path.split(file)[1]).split('_')[1]
# 		masked = GlacierSummary(file,ref_raster).mask_rasters(12,resolution)
# 		arr = masked[0]
# 		transform = masked[1]
# 		arr_dict.update({year:[arr,transform]})

# 	for shp_file in sorted(glob.glob(boundary+'*.shp')): 
# 		#if ('denali' in shp_file) or ('wrangall' in shp_file) or ('glacier' in shp_file): 
# 		uncertainty_level = make_plots.MakePlots(None,None,None).check_field(ref_raster)
# 		output_file=output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_corrected.csv'
# 		if not os.path.exists(output_file): 
# 			for k,v in arr_dict.items(): 
# 			# print(shp_file)
# 			# for file in sorted(glob.glob(raster_dir+'*full_run_vote.tif')): 
# 			# 	print(file)
# 			# 	year = (os.path.split(file)[1]).split('_')[1]
# 			# 	masked = GlacierSummary(file,ref_raster).mask_rasters(12,resolution)
# 			# 	arr = masked[0]
# 			# 	transform = masked[1]
# 				stats_df=calc_zonal_stats(v[0],shp_file,resolution,stat,f'stem_{k}',None,v[1])
# 				print(stats_df)
# 				sum_value = stats_df[f'stem_{v[0]}_{stat}'][0] #get the first value of the sum column which is the nps park sum
# 				# if sum_value == 0: 
# 				# 	break
# 				# else: 
# 				df_dict.update({year:sum_value})
# 			print(df_dict)
# 			df = pd.DataFrame(df_dict,index=[0])
# 			df.to_csv(output_file)#output_dir+os.path.split(shp_file)[1][:-4]+f'_{uncertainty_level}_updated.csv')
# 		else: 
# 			print('that file exists')