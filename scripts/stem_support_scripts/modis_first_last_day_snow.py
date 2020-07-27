# -*- coding: utf-8 -*-
"""
Created on Tues May 19 17:33:11 2020

@author: broberts (with some function inputs from jbraaten)

This script is used to collect the first and last day of snow for AK in climate regions and elevation bands. These data are then used to paramaterize the medoid composites that form the basis for the LT algorithm. This is intended to 
minmize conflation of snow and glacier ice in the AK study areas for the AK glaciers project with the NPS. 
Inputs: 
Steps: 
1. extract individual shapefiles from the AK climate regions
2. clip dem to the individual regions
3. clip modis snow metrics to the individual regions
4. mask the modis snow metrics using the elevation range masks from the dem
5. calculate zonal stats
6. write zonal stats to a df for export 
7. need to also write out some kind of geographic bounds so we know where to apply these dates
"""

import os
import sys
import pandas as pd 
from osgeo import gdal
import pyParz
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm
import matplotlib as mpl
import geopandas as gpd
#import pathlib
import glob
import matplotlib.pyplot as plt
import subprocess
import multiprocessing
from matplotlib import cm
import numpy as np
import pickle
from affine import Affine
import error_analysis as analysis 
import seaborn as sns
import string
from pathlib import Path

def clip_it(shp_file,raster_file,resolution,output_directory): 
	#for shp_file in glob.glob(climate_regions+"*.shp"):
		#print(shp_file)

	fn = os.path.split(shp_file)[1][:-4]+f'_AK_{os.path.split(raster_file)[1][:-4]}_{resolution}m.tif'
	output_filepath = output_directory+fn
	cmd = 'gdalwarp -cutline '+shp_file+ ' -crop_to_cutline -dstalpha -t_SRS "EPSG:3338" '+raster_file+' '+ output_filepath 
	#cmd_list.append(cmd)
	return cmd
def process_climate_regions(climate_regions,resolution,output_directory,input_dem,modis_directory):
	"""Read in and process the AK climate regions."""
	cmd_list = []
	#run for a directory of files
	if not modis_directory == None: 
		print('running for a directory')
		for rast_file in glob.glob(modis_directory+'*.tif'): 
			for shp_file in glob.glob(climate_regions+"*.shp"):
				cmd = clip_it(shp_file,rast_file,resolution,output_directory)
				cmd_list.append(cmd)
		print('cmd list len for multiple rasters is: ',len(cmd_list))
	else:
		for shp_file in glob.glob(climate_regions+"*.shp"):
			cmd=clip_it(shp_file,input_dem,resolution,output_directory)
			print('THE COMMAND IS: ', cmd)
			cmd_list.append(cmd)
		print('cmd_list len for one raster is: ', len(cmd_list))
	#run for a single file (eg dem)
	# for shp_file in glob.glob(climate_regions+"*.shp"):
	# 	#print(shp_file)
	# 	fn = os.path.split(shp_file)[1][:-4]+f'_AK_{modifer}_{resolution}m.tif'
	# 	output_filepath = output_directory+fn
	# 	cmd = 'gdalwarp -cutline '+shp_file+ ' -crop_to_cutline -dstalpha -t_SRS "EPSG:3338" '+input_dem+' '+ output_filepath 
	# 	cmd_list.append(cmd)
	return cmd_list

def create_masks(input_file): 
	"""Define masks from a dem between two elevations."""
	#get all of the dem files for the different climate zones
	#for file in glob.glob(dem_inputs+'*.tif'): 
	ds = gdal.Open(input_file)
	#print('file is: ',file)
	arr = np.array(ds.GetRasterBand(1).ReadAsArray())
	arr_max = int(arr.max())
	arr_min = int(arr.min())
	#print('min/max are: ',arr_min,arr_max)
	#step = (arr_max-arr_min)/10
	mask_dict = {}
	if not arr_min >= 0: 
		arr_min = 0 
	else: 
		pass
	elevs = list(range(arr_min,arr_max,250))
	
	for i in elevs: 
		#print(f'i is: {i}, starting the next elevation band')
		masked = np.where((arr>= i) & (arr<i+250),1,0)
		mask_dict.update({i:masked}) 
	ds = None	
	return mask_dict

def check_arrays(arr,mask): 
	"""Make logical checks on the size/shape of numpy arrays."""
	#check that the rows match
	if not arr.shape[0] == mask.shape[0]: 
		print('rows do not match, fixing')
		if arr.shape[0] < mask.shape[0]: #pad the difference to the bottom of the array
			arr = np.pad(arr,((0,mask.shape[0]-arr.shape[0]),(0,0)),'constant') 
		elif arr.shape[0] > mask.shape[0]: 
			mask = np.pad(mask,((0,arr.shape[0]-mask.shape[0]),(0,0)),'constant')

	#check that the cols match
	if not arr.shape[1] == mask.shape[1]: 
		#print('the arr shape is: ', arr.shape)
		#print('the mask shape is: ', mask.shape)
		#pad the cols
		print('cols do not match, fixing')
		if arr.shape[1] < mask.shape[1]: #pad the difference to the right of the array
			arr = np.pad(arr,((0,0),(0,mask.shape[1]-arr.shape[1])),'constant')
		elif arr.shape[1] > mask.shape[1]: 
			mask = np.pad(mask,((0,0),(0,arr.shape[1]-mask.shape[1])),'constant')

	# if they already match or they are now fixed then make the binary mask 
	print('making mask')
	masked_arr = arr*mask 

	return masked_arr,mask
def write_raster(raster,elev,output_directory,mod_region_id,year,ds,value): 
	"""Helper function to write a raster to disk."""
	output_file = output_directory+f'{value}_{int(elev)}m_{int(elev)+250}m_'+mod_region_id+year+'.tif'
	[cols, rows] = raster.shape #they're the same shape so just take the first one
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(output_file, rows, cols, 1, gdal.GDT_UInt16)
	outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
	outdata.SetProjection(ds.GetProjection())##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(raster)
	outdata.GetRasterBand(1).SetNoDataValue(np.nan)##if you want these values transparent
	outdata.FlushCache() ##saves to disk!!
	return None


def process_modis_data(modis_inputs,dem_inputs,output_directory,make_rasters,write_to_pickle,pickle_directory):
	"""Get out the mean last day/first day of snow by elevation band and climate zone."""
	output_list = []
	for mod_file in glob.glob(modis_inputs+'*.tif'): 
		output_dict = {}
		#print(mod_file)
		ds = gdal.Open(mod_file)
		geotransform = ds.GetGeoTransform()
		#print(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size)
		# ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
		# lrx = ulx + (ds.RasterXSize * xres)
		# lry = uly + (ds.RasterYSize * yres)
		#get some of the info we need to populate the df
		year = os.path.split(mod_file)[1].partition('AK')[2][1:5] #pull the year out of the name string. This is kind of hardcoded for the current name structure
		#print('the year is: ',year)
		mod_region_id = os.path.split(mod_file)[1].partition('AK')[0] #get the name of the region
		#use band to determine first or last day of snow in the output df
		arr_first = np.array(ds.GetRasterBand(1).ReadAsArray())
		arr_last = np.array(ds.GetRasterBand(2).ReadAsArray())
		# zeros = np.zeros(arr_first.shape)
		# print('the zeros shape is ', zeros.shape)
		# (y_index,x_index) = np.nonzero(arr_first>=0)
		# x_coords = (x_index * x_size + upper_left_x + (x_size / 2)).reshape(arr_first.shape) #add half the cell size
		# y_coords = (y_index * y_size + upper_left_y + (y_size / 2)).reshape(arr_first.shape) #to centre the point
		# print(x_coords.shape)
		# print(y_coords.shape)
		for region in glob.glob(dem_inputs+'*.tif'): #iterate through the climate region dem files 
			region_id = os.path.split(region)[1].partition('AK')[0] #get the name of the dem region 
			try: 
				if mod_region_id == region_id: 
					for k,mask in create_masks(region).items(): #k is the lower bound of the elevation zone, mask is the associated climate region/elevation band binary mask
						#get the binary elevation mask
						#check that the rows match with check_arrays and get the original array with the row/col added if needed (that's out_arr)
						first_mask_output = check_arrays(arr_first,mask) #run this to generate the first array and the simple mask and then index it
						last_mask_output =check_arrays(arr_last,mask)
						masked_first = first_mask_output[0].astype('float')
						masked_last = last_mask_output[0].astype('float')
						first_bin_mask = first_mask_output[1].astype('int16')
						last_bin_mask = last_mask_output[1].astype('int16')
						#change 0 to np.nan so it doesn't affect the mean calculation
						masked_first[masked_first == 0] = np.nan
						masked_last[masked_last == 0] = np.nan
						#print('the masked shape is: ', masked_first.shape)
						#write out a raster
						if make_rasters.lower() == 'true' and year == '2002':
							print('making rasters') 
							#first_bin_mask[first_bin_mask == np.nan] = 0
							#last_bin_mask[last_bin_mask == np.nan] = 0
							first_bin_mask[first_bin_mask == 1] = np.nanmean(masked_first)
							last_bin_mask[last_bin_mask == 1] = np.nanmean(masked_last)
							write_raster(first_bin_mask,k,output_directory,mod_region_id,year,ds,'first_day')
							write_raster(last_bin_mask,k,output_directory,mod_region_id,year,ds,'last_day')
						else:
							pass
						# outdata = None
						# band=None
						# ds=None
						#get the pixel centroid coordinates masked = np.where((arr>= i) & (arr<i+250),1,0)
						#fwd = Affine.from_gdal(*geotransform)
						#output_coords = np.argwhere(masked_first>0)
						#print(output_coords.tolist())
						#coord_list = [list(fwd*i) for i in output_coords.tolist()]
						#print(output_list)
						# x_coords = np.full(masked_first.shape,upper_left_x)
						# y_coords = np.full(masked_first.shape,upper_left_y)
						# #shift the values by res/2
						# loc_arr = np.subtract(np.arange(masked_first.shape[0]*masked_first.shape[1]).reshape(masked_first.shape),(np.full(masked_first.shape,x_size/2)))

						# bin_mask = np.where(masked_first>=0,1,0)

						# x_output_coords = bin_mask*x_coords
						# y_output_coords = bin_mask*y_coords
						# output_coords = zip(x_output_coords,y_output_coords)
						# print(output_coords)
						# print(output_coords.shape)
						#store the outputs {"type":"MultiPoint","coordinates": [[-2168447.821,921024.1641], [-2168447.921,921024.7641]]}
						output_dict = {'year':year,'climate_region':mod_region_id[:-1],'lower_bound':int(k),'upper_bound':int(k)+250,'first_day_mean':np.nanmean(masked_first),'last_day_mean':np.nanmean(masked_last)}#,
						#'geometry':'{"type":"MultiPoint","coordinates":'+f'{coord_list}'+'}'}#'upper_left_x':ulx,
						#'upper_left_y':uly,'lower_right_x':lrx,'lower_right_y':lry}
						output_list.append(output_dict)
					else: 
						pass
			except TypeError as e:
				print('that did not work but the error was: ',e)
				print(mod_region_id,region_id)
				continue

	#clean up the dataset
		ds = None
	output_df = pd.DataFrame(output_list).sort_values(by=['climate_region'])
	if write_to_pickle.lower() =='true':
		pickled_df = pickle.dump(output_df, open(pickle_directory+'modis_first_last_day_snow_by_climate_region', 'ab' ))
	return output_df

def graph_first_last_snow_days(input_df,variable): 
	df = pickle.load(open(input_df,'rb'))
	region_list = df['climate_region'].unique()
	year_list = sorted(df['year'].unique())
	rows = 3
	cols = 4
	fig,axes = plt.subplots(rows,cols,figsize=(10,10))
	axes = axes.flatten()
	palette = sns.light_palette('Navy', len(df['year'].unique()))
	for i in range(rows*cols):
		try: 
			count = 0 
			df_slice = df[df['climate_region']==region_list[i]].sort_values('year')
			for j in year_list: 
				df_slice[df_slice['year']==j].sort_values('low_bound').plot.line(x='low_bound',y=variable,ax=axes[i],legend=False,color=list(palette)[count]) #variable denotes first or last day. Can be first_day_mean or last_day_mean
				count +=1

			axes[i].set_title(string.capwords(str(region_list[i]).replace('_',' ')))
			axes[i].xaxis.label.set_visible(False)
			
		except IndexError: 
			continue

	norm = mpl.colors.Normalize(vmin=min(year_list),vmax=max(year_list))
	cmap = sns.light_palette('Navy',len(year_list),as_cmap=True)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	
	fig.subplots_adjust(bottom=0.1, top=0.9, left=0.0, right=0.7,
                    wspace=0.5, hspace=0.02)
	
	# put colorbar at desire position
	cbar_ax = fig.add_axes([0.95, 0.1, .01, .8])
	#add colorbar
	fig.colorbar(sm,ticks=np.linspace(int(min(year_list)),int(max(year_list))+1,int(len(year_list))+1),boundaries=np.arange(int(min(year_list)),int(max(year_list))+1,1),cax=cbar_ax)
	#add common x and y labels
	fig.text(0.5, 0.03, 'Lower elevation bound (m asl)', ha='center',size='large')
	fig.text(0.04, 0.5, string.capwords(variable.replace('_',' ')), va='center', rotation='vertical',size='large')
	#add common title
	plt.suptitle(f'Mean {variable.replace("_"," ").rsplit(" ", 1)[0]} of snow by elevation band and climate region')
	plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
	plt.show()
	plt.close('all')

def generate_mosaic(input_directory,output_directory,variable): 
	#output_tif = output_dest+f'AK_5m_original_mosaic_{region}_region.tif'
	dem_fps = [str(file) for file in Path(input_directory).glob('*.tif')]

	#input_files = filepath+'*.tif' #[str(file) for file in Path(filepath).glob('*.tif')]
	for year in range(2002,2020): 
		output_vrt = output_directory+f'MODIS_{variable}_{year}_by_climate_region_and_elevation_band.vrt'
		year_list = [i for i in dem_fps if str(i[-8:-4])==str(year)]
		print(year_list)
		cmd = gdal.BuildVRT(output_vrt, dem_fps, outputSRS = 'EPSG:3338', allowProjectionDifference=True,srcNodata=0) 
	#make a tif from the vrt file 
	#print('making the tif file')
	#ds = gdal.Translate(output_tif,output_vrt,format='GTiff',outputSRS=epsg)

	#close the datasets
	#ds = None
		cmd = None
def run_cmd(cmd):
	"""Helper function for running things in parallel."""
	print(cmd)  
	return subprocess.call(cmd, shell=True)

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		
		#construct variables from param file
		climate_regions = variables['climate_regions']
		modis_directory = variables['modis_directory']
		dem = variables['dem']
		input_directory = variables['input_directory']
		resolution = variables['resolution']
		output_directory = variables['output_directory']
		make_rasters = variables['make_rasters']
		write_to_pickle = variables['write_to_pickle']
		pickle_directory = variables['pickle_directory']
		variable = variables['variable']
	#for file in glob.glob(input_directory+'*.tif'): 
	

		#cmd=process_climate_regions(climate_regions,resolution,output_directory,dem,file[:-4])
		#subprocess.call(cmd,shell=True)
		#create_masks(input_directory,output_directory,modis_directory)
		df = process_modis_data(modis_directory,input_directory,output_directory,make_rasters,write_to_pickle,pickle_directory)
		#graph_first_last_snow_days(pickle_directory+'modis_first_last_day_snow_by_climate_region',variable)
		#df.to_csv(output_directory+'elevation_mean_by_climate_region_run_w_coords.csv')
		#generate_mosaic(input_directory,output_directory,variable)
		# clip_cmds= process_climate_regions(climate_regions,resolution,output_directory,dem,modis_directory)
		# # run the commands in parallel 
		# pool = multiprocessing.Pool(processes=20)
		# pool.map(run_cmd, clip_cmds)  
		# pool.close()
if __name__ == '__main__':
	main()

