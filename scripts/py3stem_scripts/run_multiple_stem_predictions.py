
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
	return subprocess.call(cmd)#, shell=True)

def make_process_cmds(file_list,predict_script): 
	cmd_list = []
	for param_file in file_list: 
		#out_file = os.path.split(raster)[1][:-4]+'_clipped.tif' #get the raster name
		cmd = ['python', '{predict_script}'.format(predict_script=predict_script), '{param_file}'.format(param_file=param_file)]   
		cmd_list.append(cmd)  
	return cmd_list

#gdalwarp -cutline INPUT.shp -crop_to_cutline -dstalpha INPUT.tif OUTPUT.tif

def make_mosaic_cmds(file_list,mosaic_script): 
	cmd_list = []
	for folder in file_list: 
		dir_name = os.path.split(folder)[1]
		cmd = ['python','{mosaic_script}'.format(mosaic_script=mosaic_script),'Byte','{dir_name}'.format(dir_name=dir_name),'vote']
		cmd_list.append(cmd)
		return cmd_list
def main(): 
	 # get the arguments
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)

		#construct variables from param file
		param_dir = variables["param_dir"]
		predict_script = variables["predict_script"]
		mosaic_script = variables['mosaic_script']
		index = variables['index']
	
	param_file_list = glob.glob(param_dir+'*prob.txt')
	# if index.lower() == 'ndsi': 
	# 	param_file_list = [x for x in param_file_list if not ('tcb' in x)] 
	print(param_file_list)

	# run the commands in parallel 
	pool = multiprocessing.Pool(processes=len(param_file_list)+1)
	pool.map(run_cmd, make_process_cmds(param_file_list,predict_script))  
	pool.close()

if __name__ == '__main__':
	main()