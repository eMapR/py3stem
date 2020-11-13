
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


def calc_zonal_stats(raster,shp,resolution,stat,source,ref_dataset,transform_info): 
	"""Calculate pixel counts inside polygons."""
	geo_df = gpd.read_file(shp)
	
	if type(raster)=='str': 
		with rasterio.open(raster) as src: 
			#construct the transform tuple in the form: top left (x coord), west-east pixel res, rotation (0.0), top left northing (y coord), rotation (0.0), north-south pixel res (-1*res)
			transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
			arr = src.read(1)#.astype('float')
			if 'stem' in source: 
				arr = np.where(arr==12,1,0)#input_arr[input_arr==12,1]
			#arr[arr == None] = 0
			#arr[arr == np.nan] = 0
	else: 
		print('using arr as raster')
		arr = raster
		transform = transform_info
	#rasterstats zonal stats produces a list of dicts, get the value
	stats = zonal_stats(geo_df,arr,stats=stat,transform=transform,nodata=255) #gets the values at the raster pixels for the given points
	output_geodf = geo_df.join(pd.DataFrame(stats))#.drop(['left','right','top','bottom'])
	#rename cols so they don't get angry when we join
	old_names = output_geodf.columns
	new_names = [source+'_'+i for i in old_names]
	column_names = dict(zip(old_names,new_names))
	output_geodf.rename(columns=column_names,inplace=True)	
	return output_geodf
	

def read_pickles(*argv): 
	predicted_file = argv[9]+f'{argv[10]}_certinaty_level_{argv[7]}_zonal_stats_df'
	print(f'predicted file is {predicted_file}')
	if (argv[8].lower()=='true') and not (os.path.exists(predicted_file)): 
		print('pickling...')
		#generate zonal stats and pickle
		predicted = calc_zonal_stats(argv[1],argv[2],argv[3],argv[4],argv[6],argv[11],argv[13]) #added None 10/12/2020 because the calc_zonal_stats was modified for other uses
		predicted_ds = pickle.dump(predicted,open(predicted_file,'ab'))
		return predicted #remove actual 
		
			
	else: #read in the pickled df if its the same data
		print('reading from pickle...')
		predicted_df = pickle.load(open(predicted_file,'rb'))
		return predicted_df #removed the actual df 

def calc_confusion_matrix(*argv):#actual_source,predicted_source,stat,): 
	"""Calculate a confusion matrix to compare nlcd or rgi and classification."""
	data = read_pickles(*argv)
	#print(data)
	actual = ref_dataset.get_csvs(argv[11],'discard','y').replace(np.nan,0)#previously getting values from raster, now getting from ref dataset csv
	#print('id col for actual is: ', actual.id)
	#print(f'shape is: {actual.shape}')
	
	predicted = data.replace(np.nan,0)
	predicted.index = np.arange(1,len(predicted)+1)
	predicted = predicted[predicted.index.isin(actual.index)] 
	actual_col = actual['binary']
	predicted_col = predicted[str(argv[6]+'_'+argv[4])]
	actual_ls = [float(i) for i in list(actual_col)]
	predicted_ls = [float(i) for i in list(predicted_col)]
	labels = sorted(list(set(list(actual_col)+list(predicted_col))))
	results = confusion_matrix(actual_ls, predicted_ls,labels) 
	
	classification_output = pd.DataFrame(classification_report(actual_ls,predicted_ls,output_dict=True)).transpose().reset_index().rename(columns={'index':'stat'})
	classification_output['model_accuracy'] = accuracy_score(actual_ls,predicted_ls)
	#print('output is: ', classification_output)
	if argv[12].lower() == 'true': 
		ax=plt.subplot()
		sns.heatmap(results,annot=True,ax=ax,fmt='g')
		ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
		ax.set_title(f'Certainty level {argv[7]} {argv[10]} model') 
		#ax.set_title(f'Northern region confusion_matrix') 
		ax.set_xticklabels(labels)
		ax.set_yticklabels(labels)
		plt.show()
		plt.close('all')
	else: 
		print('returning error report only')
	return classification_output


# def compare_error_outputs(raster_dir,modifier):
# 	output_dict = {}
# 	for file in glob.glob(raster_dir+'merge.tif'): 
# 		 with rasterio.open(file) as src: 
# 			arr = src.read(1)*100
# 			transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
# 			for x in range(50,100,5): 
# 				arr1=arr[arr>=x]


def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		#shapefile = variables["shapefile"]
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

	#generates a confusion matrix and error report 
	generate_visual = 'true'
	#def compare_error_outputs(raster_dir,modifier):
	output_dict = {}
	#print(raster_dir)
	if 'one' in uncertainty_layer: 
		certainty_level = 'one'
	elif 'two' in uncertainty_layer: 
		certainty_level = 'two'
	elif 'three' in uncertainty_layer:
		certainty_level = 'three'
	else: 
		certainty_level = 'combined'
	output_filename = pickle_dir+f'output_dict_{certainty_level}_northern_region_combined'
	if not os.path.exists(output_filename): 
		for file in sorted(glob.glob(raster_dir+'*merge.tif')):
			#print(file) 
			with rasterio.open(file) as src: 
				arr = src.read(1)#*100
				arr = arr*100
				transform = (src.bounds[0],resolution,0.0,src.bounds[3],0.0,-1*resolution)
				for x in range(50,100,5): 
					#arr=np.where(arr>=float(x))
					arr1 = np.where(arr>=float(x),1,0)#input_arr[input_arr==12,1]

					#print(arr1)
					#print(arr1.shape)
					error_stats=calc_confusion_matrix(None,arr1,random_pts,resolution,stat,str(x),predicted_source,certainty_level,write_to_pickle,pickle_dir,f'{os.path.split(file)[1][:-4]}_{x}',uncertainty_layer,generate_visual,transform)
					output_dict.update({os.path.split(file)[1][:-4]+f'_{x}':error_stats})
		print(output_dict)
		output = pickle.dump(output_dict,open(output_filename,'ab'))
		df_input=output_dict
	else: 
		print('working from pickled dictionary')
		df_input = pickle.load(open(output_filename,'rb'))
	#output_df = pd.DataFrame(df_input,index=list(df_input.keys()))
	#print(df_input)
	df_list = []
	for k,v in df_input.items(): 
		v['model'] = k
		df_list.append(v)
	final_df=pd.concat(df_list)
	print(final_df)
	csv_path = output_dir+os.path.split(output_filename)[1]+'.csv'
	if not os.path.exists(csv_path): 
		final_df.to_csv(csv_path)
	else: 
		print('The csv for this certainty level already exists')
	# t1 = datetime.now()
	# print((t1-t0)/60)
	# plot_df = {}
	# count = 0 
	# for k,v in output_dict.items(): 
	# 	plot_df.update({k:v['model_accuracy'].iloc[0]})

	# pd.DataFrame(plot_df,index=range(len(plot_df))).T.plot(kind='bar')

	# fig,ax = plt.subplots(4,4)
	# ax = ax.flatten()
	# count = 0 
	# for k,v in output_dict.items(): 
	# 	ax[count].plot(k,v[])
	# 	ax[count].set_title(f'{k}')
	#plt.show()
	#plt.close('all')
	
if __name__ == '__main__':
	main()