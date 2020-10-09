# -*- coding: utf-8 -*-
"""
Created on 10/1/2020"""

import os 
import sys
import glob 
import numpy as np 
import subprocess
import multiprocessing
import json
	
def run_cmd(cmd):
	print(cmd)  
	return subprocess.call(cmd, shell=True)

def make_process_cmds(raster_list,clip_shape): 
	cmd_list = []
	for raster in raster_list: 
		out_file = os.path.split(raster)[1][:-4]+'_clipped.tif' #get the raster name
		cmd = 'gdalwarp -cutline '+clip_shape+' -crop_to_cutline '+raster+' '+out_file   
		cmd_list.append(cmd)  
	return cmd_list

#gdalwarp -cutline INPUT.shp -crop_to_cutline -dstalpha INPUT.tif OUTPUT.tif


def main(): 
	 # get the arguments
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)

		#construct variables from param file
		raster_dir = variables["raster_dir"]
		input_shape = variables["input_shape"]
	
	raster_list = glob.glob(raster_dir+'*.tif')
	print(raster_list)
	# run the commands in parallel 
	pool = multiprocessing.Pool(processes=20)
	pool.map(run_cmd, make_process_cmds(raster_list,input_shape))  
	pool.close()

if __name__ == '__main__':
	main()