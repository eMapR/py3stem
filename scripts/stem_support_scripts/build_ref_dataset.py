"""Script for collating, cleaning and organizing the data collected for AK glaciers reference dataset."""
import glob
import sys
import os
import numpy as np 
import pandas as pd 
import json
import matplotlib.pyplot as plt
def get_csvs(csv,remove_field,remove_val): 
	df = pd.read_csv(csv)
	df.index = np.arange(1,len(df)+1)

	df1 = df[df[remove_field]!=remove_val] #remove anything that we wanted to discard in classification
	df1 = df1[df1[remove_field]!='m']
	df1 = df1[df1['class']!='u'] #remove the undecided points
	#print(df.head)
	#print(df.shape)
	#recode 
	df1['binary'] = df1['class'].map({'u': 0, 's': 0,'w':0,'n':0,'g':1,'d':1,'ha':1,'tw':1,'sh':1})
	return df1

class MakePlots(): 
	"""A class to make plots."""
	def __init__(self,input_csv,plot_fields): 

		self.input_csv=input_csv
		self.plot_fields=plot_fields
	def make_hist(self): 
		df = pd.read_csv(self.input_csv)
		#print(df.head())
		#print(self.plot_fields) 
		if 'one' in self.input_csv: 
			category = 'one'
		elif 'two' in self.input_csv: 
			category = 'two'
		elif 'three' in self.input_csv: 
			category = 'three'
		elif 'combine' in self.input_csv: 
			category = 'non_glacier'
		else: 
			print('double check your inputs')
			category = 'null'
		fig,ax = plt.subplots(1,2)
		ax = ax.flatten()
		#print(ax.shape)
		colors = ['darkblue','darkred']
		for i in range(len(self.plot_fields)): 
			# print(f'count is: {i}')
			#print(f'i is: {i}')
			# print(df[self.plot_fields[i]])
			# print(pd.Series(self.plot_fields[i]).value_counts())
			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.strip()
			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.lower()
			pd.Series(df[self.plot_fields[i]]).value_counts().plot(kind='bar',color=colors[i],ax=ax[i])
			#ax.hist(df[self.plot_fields[0]])#pd.Series(df[self.plot_fields[0]]).value_counts().plot(kind='bar',color=colors[0])
			ax[i].set_xlabel(self.plot_fields[i]);ax[i].set_ylabel('count')
			ax[i].set_title(f'Uncertainty category {category} {self.plot_fields[i]}')
			if self.plot_fields[i].lower() == 'class': 
				print('annotate')
				print(df[self.plot_fields[i]].count())
				ax[i].annotate(f'n = {df[self.plot_fields[i]].count()}',xy=(0.4,0.8),xycoords='figure fraction')
			elif self.plot_fields[i].lower() == 'image_source': 
				plt.xticks(rotation=0)
		#fig.suptitle(f'Uncertainty category {category} distributions') 
		#plt.tight_layout()
		plt.show()
		plt.close('all')
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f) 
		csv_directory = variables["csv_directory"]
		remove_field = variables['remove_field']
		remove_val	= variables['remove_val']
		plot_fields = variables['plot_fields']


		for file in glob.glob(csv_directory+'*.csv'): 
			if '2001' in file: 
				print(file)
				#input_file=get_csvs(file,remove_field,remove_val)
				plots = MakePlots(file,plot_fields)
				plots.make_hist()
			else: 
				pass
if __name__ == '__main__':
	main()