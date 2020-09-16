import os
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import json
import glob 

class MakePlots(): 
	"""A class to make plots."""
	def __init__(self,input_csv,plot_fields): 

		self.input_csv=input_csv
		self.plot_fields=plot_fields
	def make_hist(self): 
		df = pd.read_csv(self.input_csv)
		print(df.head())
		print(self.plot_fields)
		if 'one' in self.input_csv: 
			category = 'one'
		elif 'two' in self.input_csv: 
			category = 'two'
		elif 'three' in self.input_csv: 
			category = 'three'
		else: 
			print('double check your inputs')
			category = 'null'
		fig,ax = plt.subplots(1,2)
		ax = ax.flatten()
		colors = ['darkblue','darkred']
		for i in range(len(self.plot_fields)): 
			# print(f'count is: {i}')
			# print(f'i is: {i}')
			# print(df[self.plot_fields[i]])
			# print(pd.Series(self.plot_fields[i]).value_counts())
			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.strip()
			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.lower()
			ax[i]=pd.Series(df[self.plot_fields[i]]).value_counts().plot(kind='bar',color=colors[i])
			ax[i].set_xlabel(self.plot_fields[i]);ax[i].set_ylabel('count')
			ax[i].set_title(f'Uncertainty category {category} distributions') 
			if self.plot_fields[i].lower() == 'class': 
				ax[i].annotate(f'n = {df[self.plot_fields[i]].count()}',xy=(0.8,0.8),xycoords='figure fraction')
			else: 
				plt.xticks(rotation=45)
		plt.show()
		plt.close('all')
		
def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		csv_dir= variables["csv_dir"]
		#plot_fields = list(variables['plot_fields'])
		plot_fields = variables['plot_fields']
		#print(type(plot_fields))
	for file in glob.glob(csv_dir+'*.csv'): 
		plots = MakePlots(file,plot_fields)
		plots.make_hist()
	# ["class", "image_source"]   

if __name__ == '__main__':
	main()

# df = pd.read_csv(self.input_csv)
# 		df[self.plot_field] = df[self.plot_field].str.strip()
# 		df[self.plot_field] = df[self.plot_field].str.lower()
# 		ax=pd.Series(df[self.plot_field]).value_counts().plot(kind='bar',color='darkred')
# 		ax.set_xlabel(self.plot_field);ax.set_ylabel('count')
# 		ax.set_title(f'Uncertainty category {self.modifier} distributions') 
# 		if self.plot_field.lower() == 'class': 
# 			ax.annotate(f'n = {df[self.plot_field].count()}',xy=(0.8,0.8),xycoords='figure fraction')
# 		else: 
# 			plt.xticks(rotation=45)

# 		plt.show()
# 		plt.close('all')

# colors = ['darkblue','darkred']
# 		for i in range(len(self.plot_fields)): 
# 			print(f'count is: {i}')
# 			print(f'i is: {i}')
# 			print(df[self.plot_fields[i]])

# 			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.strip()
# 			df[self.plot_fields[i]] = df[self.plot_fields[i]].str.lower()
# 			ax[i]=pd.Series(self.plot_fields[i]).value_counts().plot(kind='bar',color=colors[i])
# 			ax[i].set_xlabel(self.plot_fields[i]);ax[i].set_ylabel('count')
# 			ax[i].set_title(f'Uncertainty category {category} distributions') 
# 			if self.plot_fields[i].lower() == 'class': 
# 				ax[i].annotate(f'n = {df[self.plot_fields[i]].count()}',xy=(0.8,0.8),xycoords='figure fraction')
# 			else: 
# 				plt.xticks(rotation=45)


# df = pd.read_csv(self.input_csv)
# 		if 'one' in self.input_csv: 
# 			category = 'one'
# 		elif 'two' in self.input_csv: 
# 			category = 'two'
# 		elif 'three' in self.input_csv: 
# 			category = 'three'
# 		else: 
# 			print('double check your inputs')
# 			category = 'null'
# 		fig,ax = plt.subplots(2)
# 		#ax = ax.flatten()

# 		df[self.plot_fields[0]] = df[self.plot_fields[0]].str.strip()
# 		df[self.plot_fields[0]] = df[self.plot_fields[0]].str.lower()
# 		ax[0]=pd.Series(df[self.plot_fields[0]]).value_counts().plot(kind='bar',color='darkred')
# 		ax[0].set_xlabel(self.plot_fields[0]);ax[0].set_ylabel('count')
# 		ax[0].set_title(f'Uncertainty category {category} distributions')  
# 		ax[0].annotate(f'n = {df[self.plot_fields[0]].count()}',xy=(0.8,0.8),xycoords='figure fraction')
		
# 		df[self.plot_fields[1]] = df[self.plot_fields[1]].str.strip()
# 		df[self.plot_fields[1]] = df[self.plot_fields[1]].str.lower()
# 		ax[1]=pd.Series(df[self.plot_fields[1]]).value_counts().plot(kind='bar',color='darkblue')
# 		ax[1].set_xlabel(self.plot_fields[1]);ax[1].set_ylabel('count')
# 		ax[1].set_title(f'Uncertainty category {category} distributions')  
# 		ax[1].annotate(f'n = {df[self.plot_fields[1]].count()}',xy=(0.8,0.8),xycoords='figure fraction')
# 		plt.xticks(rotation=45)
# 		plt.show()
# 		plt.close('all')

