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
def rasterize(shp,resolution,output_dest): 
	"""Convert vector to raster."""
	input_shp = ogr.Open(shp)
	shp_layer = input_shp.GetLayer()

	pixel_size = resolution
	xmin, xmax, ymin, ymax = shp_layer.GetExtent()
	head,tail = os.path.split(shp)
	output_raster = output_dest+tail[:-4]+'.tif'
	print(output_raster)
	ds = gdal.Rasterize(output_raster, shp, xRes=pixel_size, yRes=pixel_size, 
	                    burnValues=1, outputBounds=[xmin, ymin, xmax, ymax], 
	                    outputType=gdal.GDT_Byte)
	ds = None
	return output_raster

def calc_zonal_stats(raster,shp,resolution,stat,source,ref_dataset): 
	"""Calculate pixel counts inside polygons."""
	geo_df = gpd.read_file(shp)
	
	with rasterio.open(raster) as src: 
		#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
		transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
		arr = src.read(1)#.astype('float')
		#arr[arr == None] = 0
		#arr[arr == np.nan] = 0
	#rasterstats zonal stats produces a list of dicts, get the value
	stats = zonal_stats(geo_df,arr,stats=stat,transform=transform,nodata=255)
	output_geodf = geo_df.join(pd.DataFrame(stats))#.drop(['left','right','top','bottom'])
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
	if argv[8].lower()=='true': 
		print('pickling...')
		#generate zonal stats and pickle
		
		if not os.path.exists(predicted_file): 
			#print(f'creating new files {actual_file} and {predicted_file}')
			#actual = calc_zonal_stats(argv[0],argv[2],argv[3],argv[4],argv[5])
			predicted = calc_zonal_stats(argv[1],argv[2],argv[3],argv[4],argv[6],argv[11])
			#df = pd.concat([stem_df,rgi_df],axis=1)
			#actual_ds = pickle.dump(actual, open(actual_file, 'ab' ))
			predicted_ds = pickle.dump(predicted,open(predicted_file,'ab'))
			return predicted #remove actual 
		else: 
			print('both files already exist')
			pass
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
	print('id col for actual is: ', actual.id)
	print(f'shape is: {actual.shape}')
	#actual.index = np.arange(1,len(actual)+1)#actual.set_index(range(1,actual.shape[0]))
	#df.index = np.arange(1, len(df) + 1)
	#print(actual)
	#print(actual.shape)
	predicted = data.replace(np.nan,0)#read_pickles(*argv)[1].replace(np.nan,0)#calc_zonal_stats(predicted_raster,shp,resolution,stat,predicted_source)
	predicted.index = np.arange(1,len(predicted)+1)
	predicted = predicted[predicted.index.isin(actual.index)] 
	# actual.rename(columns={'2001_NLCD_classification_southern_region_id':'id'},inplace=True) predicted[predicted.set_index([range(1,predicted.shape[0])]).index.isin(predicted.set_index(['id']).index)]
	# actual = pd.merge(actual, predicted, left_on='id',how='left', indicator=True) \
 #           .query("_merge == 'left_only'") \
 #           .drop('_merge',1)
	#print('actual is:', actual['lat'],actual['lon'])
	#print(f'actual shape is {actual.shape}')
	#print(predicted.columns)
	#print('predicted is: ', predicted['2017_stem_class_one_lat'],predicted['2017_stem_class_one_lon'])
	#print(f'predicted shape is {predicted.shape}')
	#print(actual.head())
	#print(predicted.head())
	actual_col = actual['binary']#actual[str(argv[5]+'_'+argv[4])]
	predicted_col = predicted[str(argv[6]+'_'+argv[4])]
	#print(actual_col.unique())
	#print(predicted_col.unique())
	actual_ls = [float(i) for i in list(actual_col)]
	predicted_ls = [float(i) for i in list(predicted_col)]
	labels = sorted(list(set(list(actual_col)+list(predicted_col))))
	#print(labels)
	results = confusion_matrix(actual_ls, predicted_ls,labels) 
	#disp = plot_confusion_matrix(None,actual_ls,predicted_ls,display_labels=labels,cmap=plt.cm.Blues)
	#fig,(ax,ax1) = plt.subplots(nrows=1,ncols=2)
	ax=plt.subplot()
	sns.heatmap(results,annot=True,ax=ax,fmt='g')
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	ax.set_title(f'Confusion Matrix: {argv[5]} {argv[6]} {argv[7]} model run') 
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	print(results) 
	print ('Accuracy Score :',accuracy_score(actual_ls, predicted_ls))
	print ('Report : ')
	print (classification_report(actual_ls, predicted_ls))
	plt.show()
	plt.close('all')


def create_zonal_stats_df(stem_raster,rgi_raster,shp,resolution,output_dir,boundary,zoom,pickle_dir,read_from_pickle,stat): 
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
	def select_random_pts_from_raster(self,target_value):
		"""From an input raster select a random subset of points (pixels)."""

		#read in the raster
		with rasterio.open(self.dif_raster) as src: 
			arr = xr.open_rasterio(src,chunks=1000000)#1,(2917*6),(5761*6)))
			profile = src.profile

			print(arr.shape)
			if len(arr.shape) > 2: 
				try: 
					arr = arr.squeeze(axis=0)
				except: 
					arr = arr[0, :, :]

			else: 
				print('the shape of your array is: ', arr.shape)
			# print('exited if statement')
			# print(arr.shape)
			# print(type(arr))
			arr= arr.where(arr==1,drop=True) #changed so that the user can select the value you want 
			# print(arr.shape)
			# print(arr)
			new_arr = arr.stack(z=('x','y')).reset_index('z')
			new_arr = new_arr.dropna('z',how='any')
			print('passed the new_arr')
			
			# print(new_arr)
			subset_list = []
			coords_list = []
			seed(1)
		# 	print(arr)
		# # 	print(arr.values.sum())
			#indices = [9*i + x for i, x in enumerate(sorted(np.random.choice(range(0,arr.shape[0]), 500,replace=False)))]
			# if target_value == 0: 
			# 	subset = np.random.choice(range(0, new_arr.shape[0]), 34810150,replace=False)
			# 	print('created subset')
			# 	for i in subset: 
			# 		#i = int(i)
			# 		subset_list.append(tuple([float(new_arr[int(i)].coords['y'].values),float(new_arr[int(i)].coords['x'].values),float(new_arr[int(i)].coords['band'].values)]))#append a tuple of coords in the form row,col
			# 	subset_arr = np.array(subset_list)
			# 	coords_list = np.random.choice(subset_arr,500,replace=False)
			# 	print('the coords list here is: ', print(coords_list))

	
			indices = np.random.choice(range(0, new_arr.shape[0]), 500,replace=False)
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
			gdf.to_file(self.output_dir+out_filename+'.shp')#"/vol/v3/ben_ak/vector_files/glacier_outlines/revised_class_1_output_23.shp")#, driver='GeoJSON')
			gdf.to_csv(out_filename+'.csv')

			#print(gdf)
			# output_file = self.dif_raster[:-4]+'_subset_test.tif'
			# with rasterio.open(output_file, 'w', **profile) as dst: 
			# 	dst.write(arr)
		# 		# seed random number generator
		#print(coords_list[0])

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

	
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		#shapefile = variables["shapefile"]
		resolution = int(variables["resolution"])
		output_dir = variables["output_dir"]
		pickle_dir = variables["pickle_dir"]
		ref_raster = variables["ref_raster"] #previously "/vol/v3/ben_ak/raster_files/glacier_velocity/rgi_cnd_tiles_southern_region_dissolve.tif",
		classified_raster = variables["classified_raster"]
		nlcd_raster = variables["nlcd_raster"]
		#boundary = variables["boundary"]
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
	#pts = GeneratePoints(None,uncertainty_layer,output_dir)
	#print(test.pad_raster())
	#pts.reduce_raster_size()
	#output_coords = pts.select_random_pts_from_raster(0) #the extra arg can be 0 or 1 currently with 0 being select anything that isn't glacier and 1 being glacier
	#output = Polygonize(uncertainty_layer,output_dir).raster_to_polygon()
	
	#reclassify(nlcd_raster,nlcd_version,None)
	#nlcd_disagree_summary(stem_raster)
	#create_zonal_stats_df(stem_raster,rgi_raster,shapefile,resolution,output_dir,boundary,zoom,pickle_dir,write_to_pickle,stat)
	#calc_zonal_stats(nlcd_raster,random_pts,resolution,stat,'nlcd')
	#error_arr = GeneratePoints(uncertainty_layer,None,None).get_class_size()
	#make confusion matrix
	calc_confusion_matrix(None,classified_raster,random_pts,resolution,stat,actual_source,predicted_source,model_run,write_to_pickle,pickle_dir,modifier, uncertainty_layer)
	#extract_raster_pts(nlcd_raster,random_pts,resolution)
	#rasterize(shapefile,resolution,output_dir)
	t1 = datetime.now()
	print((t1-t0)/60)
if __name__ == '__main__':
	main()
#I am working
# def read_pickles(*argv): 
# 	#head,tail = os.path.split(raster_2)
# 	actual_file = argv[9]+f'{argv[5]}_{argv[10]}_zonal_stats_df'
# 	predicted_file = argv[9]+f'{argv[7]}_{argv[6]}_{argv[10]}_zonal_stats_df'
# 	if argv[8].lower()=='true': 
# 		print('pickling...')
# 		#generate zonal stats and pickle
		
# 		if not os.path.exists(predicted_file): 
# 			print(f'creating new files {actual_file} and {predicted_file}')
# 			actual = calc_zonal_stats(argv[0],argv[2],argv[3],argv[4],argv[5])
# 			predicted = calc_zonal_stats(argv[1],argv[2],argv[3],argv[4],argv[6])
# 			#df = pd.concat([stem_df,rgi_df],axis=1)
# 			actual_ds = pickle.dump(actual, open(actual_file, 'ab' ))
# 			predicted_ds = pickle.dump(predicted,open(predicted_file,'ab'))
# 			return actual, predicted 
# 		else: 
# 			print('both files already exist')
# 			pass
# 	else: #read in the pickled df if its the same data
# 		print('reading from pickle...')
# 		actual_df = pickle.load(open(actual_file,'rb'))
# 		predicted_df = pickle.load(open(predicted_file,'rb'))
# 		#print(actual_df.head())
# 		return actual_df,predicted_df
# #argv order: #0:nlcd_raster,1:stem_raster,2:random_pts,3:resolution,4:stat,5:actual_source,6:predicted_source,7:model_run,8:write_to_pickle,9:pickle_dir,10:modifier)

# def calc_confusion_matrix(*argv):#actual_source,predicted_source,stat,): 
# 	"""Calculate a confusion matrix to compare nlcd or rgi and classification."""
# 	data = read_pickles(*argv)
# 	actual = data[0].replace(np.nan,0)#read_pickles(*argv)[0].replace(np.nan,0)
# 	predicted = data[1].replace(np.nan,0)#read_pickles(*argv)[1].replace(np.nan,0)#calc_zonal_stats(predicted_raster,shp,resolution,stat,predicted_source)
# 	#print(actual.head())
# 	#print(predicted.head())
# 	actual_col = actual[str(argv[5]+'_'+argv[4])]
# 	predicted_col = predicted[str(argv[6]+'_'+argv[4])]
# 	print(actual_col.unique())
# 	print(predicted_col.unique())
# 	actual_ls = [float(i) for i in list(actual_col)]
# 	predicted_ls = [float(i) for i in list(predicted_col)]
# 	labels = sorted(list(set(list(actual_col)+list(predicted_col))))# sorted(actual[str(argv[5]+'_'+argv[4])].unique())
# 	print(labels)
# 	results = confusion_matrix(actual_ls, predicted_ls,labels) 
# 	#disp = plot_confusion_matrix(None,actual_ls,predicted_ls,display_labels=labels,cmap=plt.cm.Blues)
# 	#fig,(ax,ax1) = plt.subplots(nrows=1,ncols=2)
# 	ax=plt.subplot()
# 	sns.heatmap(results,annot=True,ax=ax,fmt='g')
# 	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
# 	ax.set_title(f'Confusion Matrix: {argv[5]} {argv[6]} {argv[7]} model run') 
# 	ax.set_xticklabels(labels)
# 	ax.set_yticklabels(labels)
# 	print(results) 
# 	print ('Accuracy Score :',accuracy_score(actual_ls, predicted_ls))
# 	print ('Report : ')
# 	print (classification_report(actual_ls, predicted_ls))
# 	plt.show()
# 	plt.close('all')


# def create_zonal_stats_df(stem_raster,rgi_raster,shp,resolution,output_dir,boundary,zoom,pickle_dir,read_from_pickle,stat): 
# 	"""A helper function for calc_zonal_stats."""
# 	head,tail = os.path.split(stem_raster)

# 	if read_from_pickle.lower()=='true': 
# 		print('pickling...')
# 		#generate zonal stats and pickle
# 		output_file = pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'
# 		if os.path.exists(output_file): 
# 			pass
# 		else: 
# 			stem_df = calc_zonal_stats(stem_raster,shp,resolution,stat,'stem')
# 			rgi_df = calc_zonal_stats(rgi_raster,shp,resolution,stat,'rgi')
# 			df = pd.concat([stem_df,rgi_df],axis=1)
# 			pickle_data=pickle.dump(df, open(output_file, 'ab' ))
# 	elif os.path.exists(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'): #read in the pickled df if its the same data
# 		print('reading from pickle...')
# 		df = pickle.load(open(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df','rb'))
# 	else: 
# 		print(pickle_dir+f'{tail[0:8]}_stem_rgi_zonal_stats_df'+'does not exist, please make sure the settings in the param file are correct')
# 	#calculate the percent error aggregated by cell (pixelwise doesn't make sense because its binary)
# 	df['pct_err'] = (((df['stem_count']-df['rgi_count'])/df['rgi_count'])*100)
# 	#rename a col to geometry because the plot function wants that
# 	df.rename(columns={'rgi_geometry':'geometry'},inplace=True)
# 	#get rid of garbage 
# 	df = df.drop(['stem_left','stem_top','stem_right','stem_bottom','stem_geometry','rgi_left','rgi_top','rgi_right','rgi_bottom'],axis=1)
# 	#select a subset by getting rid of infs
# 	df_slice = df.replace([np.inf, -np.inf],np.nan).dropna(axis=0)#df.query('stem_count!=0')#[df['stem_count']!=0 and df['rgi_count']!=0]
# 	#read in plotting shapefiles
# 	inset = gpd.read_file(boundary)
# 	background = gpd.read_file(zoom)
# 	#do the plotting 
# 	fig,ax = plt.subplots()
# 	#make the colorbar the same size as the plot
# 	divider = make_axes_locatable(ax)
# 	cax = divider.append_axes("right",size="5%",pad=0.1)
# 	left, bottom, width, height = [0.1, 0.525, 0.25, 0.25]
# 	ax1 = fig.add_axes([left, bottom, width, height])
# 	#specify discrete color ramp 
# 	cmap = mpl.colors.ListedColormap(['#005a32','#238443','#41ab5d','#78c679','#addd8e','#d9f0a3','#ffffcc',#'#ffffcc','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32',
# 	'#F2B701','#E73F74','#180600','#E68310','#912500','#CF1C90','#f23f01',
# 	'#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']) 
# 	#'#855C75','#D9AF6B','#AF6458','#736F4C','#526A83','#625377','#68855C','#9C9C5E','#A06177','#8C785D','#467378','#7C7C7C'])
# 	#'#5F4690','#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666'])#'#5D69B1','#52BCA3','#99C945','#CC61B0','#24796C','#DAA51B','#2F8AC4','#764E9F','#ED645A','#CC3A8E','#63665b'])#["#f5b460","#F1951C","#a86813", "#793200", "#004039","#006B5F", "#62BAAC","#ba6270"])#'#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'])#
# 	norm = mpl.colors.BoundaryNorm([-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap.N) 
# 	background.plot(color='lightgray',ax=ax)
# 	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax,legend=True,cax=cax,cmap=cmap,norm=norm)
# 	ax.set_title(f'{tail[0:8]} model run AK southern processing region percent error')

# 	inset.plot(color='lightgray',ax=ax1)
# 	df_slice.loc[df_slice['pct_err']<=100].plot(column='pct_err',ax=ax1,cmap=cmap,norm=norm)
# 	ax1.get_xaxis().set_visible(False)
# 	ax1.get_yaxis().set_visible(False)
# 	plt.tight_layout() 
# 	plt.show()
# 	plt.close('all')
# def select_random_pts_from_raster(self,target_value):
# 		"""From an input raster select a random subset of points (pixels)."""

# 		#read in the raster
# 		with rasterio.open(self.dif_raster) as src: 
# 			arr = xr.open_rasterio(src,chunks=10000)#1,(2917*6),(5761*6)))
# 			profile = src.profile

# 			print(arr.shape)
# 			if len(arr.shape) > 2: 
# 				arr = arr.squeeze(axis=0)
# 			else: 
# 				print('the shape of your array is: ', arr.shape)
# 			# print('exited if statement')
# 			print(arr.shape)
# 			print(type(arr))
# 			arr= arr.where(arr==1,drop=True)
			
# 			print(arr)
# 			new_arr = arr.stack(z=('x','y')).reset_index('z')
# 			new_arr = new_arr.dropna('z',how='any')
# 			#new_arr = new_arr.
# 			print('got below stack function')
# 			#new_arr = new_arr[new_arr.notnull()]
# 		# 	#print(arr.shape)
# 		# 	#arr = arr.where(arr==1,drop=True)
# 			print(new_arr)

# 			coords_list = []
# 			seed(1)
# 		# 	print(arr)
# 		# # 	print(arr.values.sum())
# 			#indices = [9*i + x for i, x in enumerate(sorted(np.random.choice(range(0,arr.shape[0]), 500,replace=False)))]
# 			indices = np.random.choice(range(0, new_arr.shape[0]), 500,replace=False)
# 			#x_vals = np.random.choice(x_arr,50)#range(0, arr.shape[0]), 500)
# 			#y_vals = np.random.choice(y_arr,50)
# 			#print(indices)
# 		# 	print(arr)
# 		# 	#rows = random.sample(range(0, arr.shape[0]), 100)
# 		# 	#for row,col in zip(rows,cols):
# 			for i in indices: 
# 				print(i,f'i is type {type(i)}')
# 				#print(arr[row,col].coords['band'].values)
# 				i = int(i)
# 				coords_list.append(tuple([float(new_arr[i].coords['y'].values),float(new_arr[i].coords['x'].values),float(new_arr[i].coords['band'].values)]))#append a tuple of coords in the form row,col
# 				#coords_list.append(tuple([float(arr[row,col].coords['y'].values),float(arr[row,col].coords['x'].values),float(arr[row,col].coords['band'].values)]))#append a tuple of coords in the form row,col
# 			df = pd.DataFrame(coords_list,columns=['lat','lon','value'])#{'lat':y_vals.tolist(),'lon':x_vals.tolist()},index=range(1,51))
# 			gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat))
# 			gdf.crs='EPSG:3338'
# 			gdf['id'] = range(1,gdf.shape[0]+1)
# 			#write out 
# 			out_filename = os.path.split(self.dif_raster)[1][:-4]
# 			#write a shapefile
# 			gdf.to_file(self.output_dir+out_filename+'.shp')#"/vol/v3/ben_ak/vector_files/glacier_outlines/revised_class_1_output_23.shp")#, driver='GeoJSON')
# 			gdf.to_csv(out_filename+'.csv')

# 			#print(gdf)
			# output_file = self.dif_raster[:-4]+'_subset_test.tif'
			# with rasterio.open(output_file, 'w', **profile) as dst: 
			# 	dst.write(arr)
		# 		# seed random number generator
		#print(coords_list[0])




# mask = None
		# with rasterio.Env():
		# 	with rasterio.open(self.input_raster) as src:
		# 		image = src.read(1).astype('int32') # first band
		# 		print(image.dtype)
		# 		results = (
		# 		{'properties': {'raster_val': v}, 'geometry': s}
		# 		for i, (s, v) 
		# 		in enumerate(
		# 			shapes(image, mask=mask, transform=src.transform)))

		# 		geoms = list(results)
		#  # first feature
		# print (geoms[0])
		# geoms = list(results)
		# gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)
		# print(gpd_polygonized_raster)

# this allows GDAL to throw Python Exceptions
# 		gdal.UseExceptions()

# 		#
# 		#  get raster datasource
# 		#
# 		src_ds = gdal.Open(self.input_raster)
# 		print(src_ds)
# 		arr = np.array(src_ds.GetRasterBand(1).ReadAsArray()).astype('float32')
# 		arr[arr != 1] = np.nan
# 		print(arr)
# 		print(arr.shape)
# 		if src_ds is None:
# 		    print ('Unable to open %s' % src_filename)
# 		    sys.exit(1)

# 		try:
# 		    srcband = src_ds.GetRasterBand(1)

# 		except RuntimeError as e:
# 		    # for example, try GetRasterBand(10)
# 		   # print ('Band ( %i ) not found' % band_num)
# 		    print (e)
# 		    sys.exit(1)

# 		#
# 		#  create output datasource
# 		#layer = ds.CreateLayer(str(layer_name), spat_ref, ogr.wkbPolygon)


# #segimg=glob.glob('Poly.tif')[0]
# #src_ds = gdal.Open(segimg, GA_ReadOnly )
# #srcband=src_ds.GetRasterBand(1)
# #myarray=srcband.ReadAsArray() 
# #these lines use gdal to import an image. 'myarray' can be any numpy array

# 		mypoly=[]
# 		for vec in rasterio.features.shapes(arr):
# 		    mypoly.append(shape(vec[0])) #shape(vec[0])
# 		#from shapely.geometry import shape
# 		geom = []#[shape(i) for i in mypoly['geometry']]
# 		for i in mypoly: 
# 			i['geometry']=
# 		#print shape(geoms[0]['geometry'])
# 		print(mypoly[0])
# 		gpd_polygonized_raster = gpd.GeoDataFrame.from_features(mypoly)
# 		print(gpd_polygonized_raster.head())
# 		# tif_drv = gdal.GetDriverByName('GTiff') # create driver for writing geotiff file
# 		# outRaster = tif_drv.CreateCopy(str(self.input_raster[:-4]), self.input_raster , 0 )# create new copy of inut raster on disk
# 		# newBand = outRaster.GetRasterBand(1)  
# 		# poly_raster = newBand.WriteArray(arr) # write array data to this band 

# 		# dst_layername = self.output_dir+"polygonized_test_2"
# 		# drv = ogr.GetDriverByName("ESRI Shapefile")
# 		# dst_ds = drv.CreateDataSource( dst_layername + ".shp" )
# 		# dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )

# 		# gdal.Polygonize(poly_raster, None, dst_layer, -1, [], callback=None )
# 		# print('finished

#print(arr)
			#arr_df = arr.to_dataframe('results').dropna('any')
			#rint(arr_df)
			#arr = arr.dropna(dim='x')
			#new_arr = arr_df.to_xarray()
			#print(new_arr)
			#print(arr_df.shape)
			#arr = arr.stack(z=('x', 'y'))
			#print(type(arr.coords))
			#print(arr.coords['x'][0])
			#band_arr = np.array(arr.coords['band'])
			#x_arr = np.array(arr.coords['x'])
			#y_arr = np.array(arr.coords['y'])
			#print(x_arr)
			#new_arr = np.stack(((arr.coords['y']),(arr.coords['x'])))
			#print(f'x arr shape is {x_arr.shape}')
			#print(f'y arr shape is {y_arr.shape}')
			#print(f'band shape is {band_arr.shape}')
			#print(x_arr)
			#new_arr = np.concatenate((y_arr,x_arr),axis=0)
			#print(new_arr.shape)
			#new_arr = arr.to_stacked_array('z')
			#print('made stacked array')
			#arr = arr.flatten()
			# print(f'here the arr shape is {arr.shape}')
			# arr = arr.reset_index('z')#.drop(['x', 'y'])
			#arr = arr.where(arr == 1, drop=True)

			#arr = arr[arr[:,:]==1]
			# print(arr.shape)
			#above here broken af
			#numpy test
			# arr = src.read()
			# profile = src.profile
			# arr = arr[arr == 1]
			# #arr = arr.where((arr==1,arr==1),drop=True)
			# #arr =arr[arr[:,0]==1, :]
			# #arr = arr[arr[:,:] == 1]

			# print(arr.shape)
			#arr = arr.squeeze(axis=0)
			#print(arr.shape)
			#arr = arr.flatten()
			#print(arr.shape)
			#print(arr.shape)
		# 	arr = xr.open_rasterio(src,(1,(2917*6),(5761*6)))
		# 	no_data=float((arr.attrs['nodatavals'])[0])
		# 	print(arr.shape)
		# 	#print(arr.chunks)

		# 	if len(arr.shape) > 2:
		# 	#remove the first axis 
		# 		arr = np.squeeze(arr,axis=0)
		# 	else: 
		# 		print('The shape of your array is: ', arr.shape)
		# 	#get some random points	to make coords indexers 
		# 	#arr = arr.where(arr == 1, drop=True)
		# 	#print(arr.shape)
		# 	#print(arr.chunks)
			#arr = arr.stack(dim_0=('x', 'y'))

			#arr = arr.reset_index('dim_0').drop(['x', 'y'])