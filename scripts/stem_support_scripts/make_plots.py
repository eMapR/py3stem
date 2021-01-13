import os
import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import json
import glob 
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import error_analysis as ea
import pickle
import geopandas as gpd
import fiona
from shapely.geometry import shape
import rasterio.features
import matplotlib as mpl
class MakePlots(): 
	"""A class to make plots."""
	def __init__(self,input_csv,plot_fields,file_list1,file_list2,file_list3): 

		self.input_csv=input_csv
		self.plot_fields=plot_fields
		self.file_list1 = file_list1
		self.file_list2 = file_list2
		self.file_list3 = file_list3
		
	def check_field(self,filename): 
		print(f'filename is: {filename}')
		try: 
			if filename == None: 
				file='null_value'

			elif self.input_csv ==None: 
				self.input_csv = 'null_value'
			if ('one' in self.input_csv) or ('one' in filename): 
				category = 'one'
			elif ('two' in self.input_csv) or ('two' in filename): 
				category = 'two'
			elif ('three' in self.input_csv) or ('three' in filename): 
				category = 'three'
			else: 
				#print('double check your inputs')
				category = 'null_value'
			return category
		except: 
			pass
	def make_hist(self): 
		df = pd.read_csv(self.input_csv)
		print(df.head())
		print(self.plot_fields)
		category = self.check_field(self.input_csv,None)
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

	def make_line(self,nrows,ncols): 
		
		uncertainty_dict = {'one':'low','two':'medium','three':'high'}
		fig,ax = plt.subplots(nrows,ncols,constrained_layout=True,sharex=True)
		if nrows*ncols > 1: 
			ax = ax.flatten()
		count = 0
		for file1,file2 in zip(self.file_list1,self.file_list2):#,self.file_list3): #list of file paths
			try: 
				print(file1)
				print(file2)
				#print(file3)
				df1 = pd.read_csv(file1)
				df2 = pd.read_csv(file2)
				#df3 = pd.read_csv(file3)
				#print(df)
				filename = os.path.split(file1)[1]
				#print(f'the filename here is now: {filename} and it is type {type(filename)}')
				category = 'null_value'#self.check_field(filename) #changed 11/16/2020- this thing is way too hard coded
				# if not (category in uncertainty_dict.keys()) or (category in uncertainty_dict.values()): #convert the pixel counts to areas for that national parks
				# 	print('converting to meters')
				# 	df1 = df1*900
				# 	df2 = df2*900
				# 	df3 = df3*900
				# 	#print('scaled df is: ',df1)
				# else: 
				# 	pass
				# #change all values into square kms instead of square meters
				# print('converting to kms')
				# df1 = df1/1000000
				# df2 = df2/1000000
				# df3 = df3/1000000
				for column1,column2 in zip(df1.columns,df2.columns):#,df3.columns): #get rid of non-year cols 
					try: 
						int(column1)
					except ValueError:
						print('col was not a year col, dropping')
						df1=df1.drop(columns=[column1])
						df2=df2.drop(columns=[column2])
						#df3=df3.drop(columns=[column3])
				plot_df1 = df1.T.reset_index().rename(columns={'index':'year',0:'area'})
				plot_df2 = df2.T.reset_index().rename(columns={'index':'year',0:'area'})
				#plot_df3 = df3.T.reset_index().rename(columns={'index':'year',0:'area'})
				#plot_df['plus_two_std'] = plot_df.area+(plot_df.area.std()*2)
				#plot_df['minus_two_std'] = plot_df.area-(plot_df.area.std()*2) 
				plots_combined = pd.concat([plot_df1,plot_df2])#,plot_df3])
				sns.lineplot(data=plots_combined,x='year',y='area',ax=ax,color='#0c2b66') #changed 11/17/2020 to make one plot for agu
				#sns.lineplot(data=plots_combined,x='year',y='area',ax=ax[count],color='darkblue')#,label='NDSI NLCD 2001',legend=False)#.plot(df.T) #changed to plot them all with CI 11/12/2020
				ax.set_ylim(150,200)
				# sns.lineplot(data=plot_df1,x='year',y='area',ax=ax[count],color='darkred',label='NDSI NLCD 2001',legend=False)#.plot(df.T)
				# sns.lineplot(data=plot_df2,x='year',y='area',ax=ax[count],color='darkblue',label='TCB NLCD 2001',legend=False)#.plot(df.T)
				# sns.lineplot(data=plot_df3,x='year',y='area',ax=ax[count],color='green',label='NDSI NLCD 2016',legend=False)#.plot(df.T)

				#ax[count].fill_between(plot_df['year'], plot_df['plus_two_std'], plot_df['minus_two_std'], color='darkred', alpha=.1)
				ax.tick_params(axis='x', rotation=90)
				ax.set_xlabel(' ')
				# ax[count].tick_params(axis='x', rotation=90)
				# ax[count].set_ylabel(' ')
				#ax[count].set_xlabel(' ')

				#ax[count].set_ylabel('Glacier covered area (sqare km')
				if not category == 'null_value':
					ax[count].set_title(f'{(uncertainty_dict[str(category)]).title()} level of certainty')
				else: 
					ax.set_title('Northern region high certainty area')
					#ax.set_title(" ".join(filename[:-4].split('_')[:-2]).title())
					ax.set_ylabel('Glacier covered area (square km)')
				#fig.text(0.002, 0.5, 'Glacier covered area (square km)', va='center', rotation='vertical',fontsize=10)
				#plt.setp(ax[:, 0], ylabel='Glacier covered area (square km)')

				count += 1 
			except IndexError: 
				print('IndexError')
				continue 
			#ax[count].set_xticks(rotation=90)
		#plt.tight_layout()
		#plt.legend()
		#fig.suptitle('High certainty glacier covered areas', fontsize=14) #hardcoded, needs to be changed
		#ax[nrows-1].legend() #add a legend
		plt.show()
		plt.close('all')

	def summarize_area_diff(self,output_dir): 
		dict_2001 = {}
		dict_2019 = {}
		for file1,file2,file3 in zip(self.file_list1,self.file_list2,self.file_list3):
			park_name = os.path.split(file1)[1][:-4]
			df1 = pd.read_csv(file1)
			df2 = pd.read_csv(file2)
			df3 = pd.read_csv(file3)
			df1.iloc[0]
			df2.iloc[0]
			df3.iloc[0]
			print(df1,df2,df3)
			for column1,column2,column3 in zip(df1.columns,df2.columns,df3.columns): #get rid of non-year cols 
					try: 
						int(column1)
					except ValueError:
						print('col was not a year col, dropping')
						df1=df1.drop(columns=[column1])
						df2=df2.drop(columns=[column2])
						df3=df3.drop(columns=[column3])
				#plot_df = df.T.reset_index().rename(columns={'index':'year',0:'area'})
			df1 = (df1*900)/1000000 #convert to square kms
			df2 = (df2*900)/1000000
			df3 = (df3*900)/1000000
			dict_2001.update({park_name:[df1['2001'].iloc[0],df2['2001'].iloc[0],df3['2001'].iloc[0]]})
			dict_2019.update({park_name:[df1['2019'].iloc[0],df2['2019'].iloc[0],df3['2019'].iloc[0]]})
		output_df_2001 = pd.DataFrame(dict_2001)
		output_df_2019 = pd.DataFrame(dict_2019)
		output_df_2001.loc['total']= output_df_2001.sum()
		output_df_2019.loc['total']= output_df_2019.sum()
		output_df_2001.to_csv(output_dir+f'nps_2001_{park_name.split("_")[-1]}.csv')
		output_df_2019.to_csv(output_dir+f'nps_2019_{park_name.split("_")[-1]}.csv')
		print(output_df_2019)

		
	def plot_area_proportions(self,nrows,ncols):
		fig,ax = plt.subplots(nrows,ncols)
		df1 = pd.read_csv(self.file_list1[0],skipfooter=1,index_col=0)#read in except the last line which is totals
		#df2 = pd.read_csv(self.file_list1,skipfooter=1)
		#df3 = pd.read_csv(self.file_list1,skipfooter=1)
		ax = ax.flatten()
		for i in range(0,(nrows*ncols)): 
			ax[i].pie(df1.iloc[:,i],labels=df1.index)
			plot_title =  " ".join(df1.columns[i].split('_')[:-2]).title()#df1.columns[i].replace("_", " ")#[' '.join(x.split('_')) for x in df1.columns[i]]#[x.replace('_', ' ') for x in df1.columns[i]]
			ax[i].set_title(plot_title)

		fig.suptitle('Year 2001 model TCB NLCD 2001', fontsize=14) #hardcoded, needs to be changed
		#plt.legend()
		plt.tight_layout()
		plt.show()
		plt.close('all')
def make_single_line_plot(input_csv): 
	df = pd.read_csv(input_csv)
	fig,ax = plt.subplots(1,1,constrained_layout=True,sharex=True)
	for column in df.columns:
		try: 
			int(column)
		except ValueError:
			print('col was not a year col, dropping')
			df=df.drop(columns=[column])
	plot_df = df.T.reset_index().rename(columns={'index':'year',0:'area'})
	sns.lineplot(data=plot_df,x='year',y='area',ax=ax,color='#0c2b66') #changed 11/17/2020 to make one plot for agu
	ax.tick_params(axis='x', rotation=90)
	ax.set_xlabel(' ')
	ax.set_title('Northern region high certainty area')
	#ax.set_title(" ".join(filename[:-4].split('_')[:-2]).title())
	ax.set_ylabel('Glacier covered area (square km)')
	plt.show()
	plt.close('all')
def make_single_heatmap(input_data): 
	ax=plt.subplot()
	sns.heatmap(input_data,annot=True,ax=ax,fmt='g',vmin=0.0,vmax=400.0,cmap='Greys')
	#ax.collections[0].colorbar.ax.set_ylim(0,400)

	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
	ax.set_title(f'Northern region low/high certainty') 
	ax.set_xticklabels(['0','1'])
	ax.set_yticklabels(['0','1'])
	plt.show()
	plt.close('all')

def plot_nps_figs(input_csv): 
	fig,ax = plt.subplots(1,1)
	#ax = ax.flatten()
	count = 0 
	# files_list = []
	# for file in csv_list: 
	# 	if ('kenai' in file) or ('katmai' in file) or ('lake_clark' in file): 
		
	df=pd.read_csv(input_csv).astype('float')#.drop(columns=['model'],axis=1)#.T.reset_index()
	#df = (df*900)/1000000 #convert to square kms comment out because conversion was done in previous script 12/16/2020
	print(df)
	#files_list.append(df)
	df.drop(columns=['Unnamed: 0'],inplace=True) #hardcoded- should change or it will yield a key error with a differnt format 
	#	else: 
	#		print('We are not interested in that park right now')
	#plot_df = pd.concat(files_list,axis=1)
	df = df.T.reset_index()
	#print(plot_df)
	df=pd.melt(df, id_vars=['index'], value_vars=[0,1,2])
	#print(plot_df)
	# df = df.T.reset_index() #transpose and make the years into a column not index
			
			# df=pd.melt(df, id_vars=['index'], value_vars=[0,1,2]) #this is hardcoded- stack the columns into one
	df.drop(columns=['variable'],inplace=True) #get rid of the variable column which is produced by the melt function 
	df=df.rename(columns={'index':'year'})
	# try: 
	# 	print(df.columns)
	# 	for column in df.columns: 
	# 		if 'Unnamed' in column: 
	# 			df.drop(columns=[column],inplace=True)
	# except KeyError: 
	# 	pass
	sns.lineplot(data=df,x='year',y='value',ax=ax,dashes=False,color='darkblue')
	#set titles and figure options
	ax.grid(which='major',linewidth=0.25,alpha=0.75)
	# title_dict = {'kenai':'Kenai Fjords National Park', 'katmai':'Katmai National Park and Preserve', 'lake_clark':'Lake Clark National Park and Preserve'}
	# if 'kenai' in file: 
	# 	fig_title = title_dict['kenai']
	# elif 'katmai' in file: 
	# 	fig_title = title_dict['katmai']
	# elif 'lake_clark' in file: 
	# 	fig_title = title_dict['lake_clark']
	ax.set_title('Southern region change all classes',fontsize='x-large')
	#ax[count].set_title((os.path.split(file)[1][:-4]).split('med')[0].replace('_',' ').title(),fontsize='x-large') #this is garbage
	ax.set_ylabel('Glacier covered area (km sq)',fontsize='x-large') #set the y axis title on only the first figure
	#ax[1].set_ylabel(' ')
	#ax[2].set_ylabel(' ')

	#ax[0].set_xlabel(' ')
	ax.set_xlabel('Image composite end year',fontsize='x-large')
	#ax[2].set_xlabel(' ')
	ax.tick_params(axis='x', rotation=45)

	#		count +=1
	plt.show()
	plt.close('all')
	
def make_percent_change_map(start_raster,end_raster,shp,resolution,output_dir,boundary,zoom,pickle_dir,read_from_pickle,stat,class_code): 

	"""A helper function for calc_zonal_stats."""

	head,tail = os.path.split(start_raster)
	output_file = pickle_dir+f'{tail}_start_year_zonal_stats_df'
	if (read_from_pickle.lower()=='true') and not (os.path.exists(output_file)): 
		print('pickling...')
		#generate zonal stats and pickle
		start_df = ea.calc_zonal_stats(start_raster,shp,resolution,stat,'start',None,class_code)
		end_df = ea.calc_zonal_stats(end_raster,shp,resolution,stat,'end',None,class_code)
		df = pd.concat([start_df,end_df],axis=1)
		pickle_data=pickle.dump(df, open(output_file, 'ab' ))
	elif os.path.exists(output_file): #read in the pickled df if its the same data
		print('reading from pickle...')
		df = pickle.load(open(pickle_dir+f'{tail}_start_year_zonal_stats_df','rb'))
	else: 
		print(pickle_dir+f'{tail}_start_year_zonal_stats_df'+'does not exist, please make sure the settings in the param file are correct')
	#calculate the percent error aggregated by cell (pixelwise doesn't make sense because its binary)
	df['pct_change'] = (((df[f'end_{stat}']-df[f'start_{stat}'])/df[f'end_{stat}'])*100)
	#areal change
	df['area_change'] = ((df[f'end_{stat}']-df[f'start_{stat}'])*float(resolution)*float(resolution)/1000000)
	#rename a col to geometry because the plot function wants that
	df.rename(columns={'end_geometry':'geometry'},inplace=True)
	#get rid of garbage -commented out 12/15/2020
	#df = df.drop(['stem_left','stem_top','stem_right','stem_bottom','stem_geometry','rgi_left','rgi_top','rgi_right','rgi_bottom'],axis=1)
	#select a subset by getting rid of infs
	#df_slice = df.replace([np.inf, -np.inf],np.nan).dropna(axis=0)#df.query('stem_count!=0')#[df['stem_count']!=0 and df['rgi_count']!=0]
	#read in plotting shapefiles
	#inset = gpd.read_file(boundary)
	background = gpd.read_file(zoom)
	#do the plotting 
	fig,ax = plt.subplots()
	#make the colorbar the same size as the plot
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right",size="5%",pad=0.1)
	left, bottom, width, height = [0.1, 0.525, 0.25, 0.25]
	#ax1 = fig.add_axes([left, bottom, width, height])
	#specify discrete color ramp 
	# cmap = mpl.colors.ListedColormap(['#005a32','#238443','#41ab5d','#78c679','#addd8e','#d9f0a3','#ffffcc',#'#ffffcc','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32',
	# '#F2B701','#E73F74','#180600','#E68310','#912500','#CF1C90','#f23f01',
	# '#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']) 
	cmap = mpl.colors.ListedColormap(['#4c0000','#682929','#773e3e','#855252','#b09090','#f7f7f7','#e2e2ed',#'#ffffcc','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32',
	'#cecee3','#b9b9d9','#a5a5cf','#9090c5','#7b7bbb','#6767b1','#5252a7',
	'#3e3e9d','#292993','#151589','#00007f','#00004c']) 
	#'#855C75','#D9AF6B','#AF6458','#736F4C','#526A83','#625377','#68855C','#9C9C5E','#A06177','#8C785D','#467378','#7C7C7C'])
	#'#5F4690','#1D6996','#38A6A5','#0F8554','#73AF48','#EDAD08','#E17C05','#CC503E','#94346E','#6F4070','#994E95','#666666'])#'#5D69B1','#52BCA3','#99C945','#CC61B0','#24796C','#DAA51B','#2F8AC4','#764E9F','#ED645A','#CC3A8E','#63665b'])#["#f5b460","#F1951C","#a86813", "#793200", "#004039","#006B5F", "#62BAAC","#ba6270"])#'#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'])#
#	norm = mpl.colors.BoundaryNorm([-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],cmap.N) 
	norm = mpl.colors.BoundaryNorm([-250,-200,-150,-100,-50,0,50,100,150,200,250,300,350,400,450,500,550,600,650],cmap.N) 

	background.plot(color='darkgray',ax=ax)
	#df_slice.loc[df_slice['pct_change']<=100].plot(column='pct_change',ax=ax,legend=True,cmap='seismic')#,norm=norm)
	df.plot(column='area_change',ax=ax,legend=True,cmap=cmap,norm=norm,cax=cax)
	ax.set_title(f'2001-2019 no low glacier probablity class areal change (sq km)')

	#inset.plot(color='lightgray',ax=ax1)
	# df_slice.loc[df_slice['pct_change']<=100].plot(column='pct_change',ax=ax1,cmap=cmap,norm=norm)
	# ax1.get_xaxis().set_visible(False)
	# ax1.get_yaxis().set_visible(False)
	plt.tight_layout() 
	plt.show()
	plt.close('all')

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		csv_dir= variables["csv_dir"]
		#plot_fields = list(variables['plot_fields'])
		plot_fields = variables['plot_fields']
		output_dir = variables['output_dir']
		input_csv = variables['input_csv']
		start_raster=variables['start_raster']
		end_raster=variables['end_raster']
		shapefile=variables['shapefile']
		resolution=variables['resolution']
		zoom = variables['zoom']
		pickle_dir=variables['pickle_dir']
		#write_to_pickle=variables['write_to_pickle']
		stat=variables['stat']

		#print(type(plot_fields))
	# for file in glob.glob(csv_dir+'*.csv'): 
	# 	plots = MakePlots(file,plot_fields,None)
	# 	plots.make_hist()
	#make_percent_change_map(start_raster,end_raster,shapefile,resolution,output_dir,None,zoom,pickle_dir,'true',stat,3) #changed the boundary thing to none 12/15/2020

	#regional_list = []
	#np_list = []
	#make both regional and national park plots
	# for file in glob.glob(csv_dir+'*one_.csv'): 
	# 	print(f'file is: {file}')
	# 	category = MakePlots(file,None,None).check_field('null_value') #tells us if this is regional or nps
	# 	print(f'category is: {category}')
	# 	if not category == 'null_value': 
	# 		regional_list.append(file)
	# 	else: 
	# 		np_list.append(file)
	# print(np_list)
	# ["class", "image_source"]   
	# print('regional list is', regional_list)
	# print('np list is', np_list)
	# MakePlots(None,None,regional_list).make_line(1,3) #just plots for the three certainty levels
	
	#make_single_line_plot(input_csv)
	################################################################################################
	#get the summary csvs- you need to change the number that you want for the certainty level 
	# ndsi_nlcd_2001 = glob.glob(csv_dir+'*combined_certainty_areas_2001_nlcd.csv')
	# ndsi_nlcd_2016 = glob.glob(csv_dir+'*high_certainty_area_2016_nlcd.csv')
	# tcb_nlcd_2001 = glob.glob(csv_dir+'*high_certainty_area_2001_nlcd_tcb.csv')
	
	#use if you want to generate national park area totals
	#ndsi_nlcd_2001 = sorted([x for x in ndsi_nlcd_2001 if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)]) #[i for i in sents if not ('@$\t' in i or '#' in i)]
	#ndsi_nlcd_2016 = sorted([x for x in ndsi_nlcd_2016 if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)]) 
	#tcb_nlcd_2001 = sorted([x for x in tcb_nlcd_2001 if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)])
	
	#use to make plots of area for nps for different models
	#MakePlots(None,None,tcb_nlcd_2001,ndsi_nlcd_2016,None).make_line(1,1)
	#MakePlots(None,None,np_list).make_line(3,3)
	#nps_files = glob.glob(csv_dir+'*.csv') #this is set up to take a directory of only relevant csvs
	plot_nps_figs(input_csv)

	#####################################################################
	#make pie charts comparing area of different certainty levels
	# input_list = glob.glob(csv_dir+'*tcb.csv')
	# print(input_list)
	# MakePlots(None,None,input_list,None,None).plot_area_proportions(3,3)
	#####################################################################
	#make csvs that summarize the output stats from different model runs
	# one = glob.glob(csv_dir+'*one_tcb.csv')
	# two = glob.glob(csv_dir+'*two_tcb.csv')
	# three = glob.glob(csv_dir+'*three_tcb.csv')
	# one = sorted([x for x in one if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)]) 
	# two = sorted([x for x in two if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)]) 
	# three = sorted([x for x in three if not ('alagnak' in x or 'aniakchak' in x or 'katmai_national_preserve' in x or 'glacier_bay_national_preserve' in x)]) 

	# MakePlots(None,None,one,two,three).summarize_area_diff(output_dir)
	######################################################################
	#make a user defined heatmap
	#input_arr = np.array([[188,23],[12,209]])
	#make_single_heatmap(input_arr)
	######################################################################
	#make maps showing percent change between two dates

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

# def make_line(self,nrows,ncols): 
		
# 		uncertainty_dict = {'one':'low','two':'medium','three':'high'}
# 		fig,ax = plt.subplots(nrows,ncols,constrained_layout=True,sharex=True)

# 		ax = ax.flatten()
# 		count = 0
# 		for file in self.file_list: #list of file paths
# 			print(f'file is: {file}')
# 			if ('alagnak' in file) or ('aniakchak' in file) or ('katmai_national_preserve' in file) or ('glacier_bay_national_preserve' in file): 
# 				print('No stats or no glaciers for that park, skipping')
# 			else: 	
# 				try: 
# 					df = pd.read_csv(file)
# 					#print(df)
# 					filename = os.path.split(file)[1]
# 					#print(f'the filename here is now: {filename} and it is type {type(filename)}')
# 					category = 'null_value'#self.check_field(filename)
# 					if not category in uncertainty_dict.keys(): #convert the pixel counts to areas for that national parks
# 						df = df*900
# 						print('scaled df is: ',df)
# 					else: 
# 						pass
# 					#change all values into square kms instead of square meters
# 					df = df/1000000
# 					#print('df is now: ', df)
# 					#print(df.T)
# 					#print(f'the function category is: {category}')
# 					for column in df.columns: #get rid of non-year cols 
# 						try: 
# 							int(column)
# 						except ValueError:
# 							print('col was not a year col, dropping')
# 							df=df.drop(columns=[column])
# 					plot_df = df.T.reset_index().rename(columns={'index':'year',0:'area'})
# 					#plot_df['plus_two_std'] = plot_df.area+(plot_df.area.std()*2)
# 					#plot_df['minus_two_std'] = plot_df.area-(plot_df.area.std()*2) 
# 					sns.lineplot(data=plot_df,x='year',y='area',ax=ax[count],color='darkred')#.plot(df.T)
# 					#ax[count].fill_between(plot_df['year'], plot_df['plus_two_std'], plot_df['minus_two_std'], color='darkred', alpha=.1)
# 					ax[count].tick_params(axis='x', rotation=90)
# 					ax[count].set_ylabel(' ')
# 					ax[count].set_xlabel(' ')

# 					#ax[count].set_ylabel('Glacier covered area (sqare km')
# 					if not category == 'null_value':
# 						ax[count].set_title(f'{(uncertainty_dict[str(category)]).title()} level of certainty')
# 					else: # " ".join(["John", "Charles", "Smith"])
# 						ax[count].set_title(" ".join(filename[:-4].split('_')).title())
# 					fig.text(0.001, 0.5, 'Glacier covered area (square km)', va='center', rotation='vertical')
# 					#plt.setp(ax[:, 0], ylabel='Glacier covered area (square km)')


# 					count += 1
# 				except IndexError: 
# 					print('IndexError')
# 					continue 
# 			#ax[count].set_xticks(rotation=90)
# 		#plt.tight_layout()
# 		plt.show()
# 		plt.close('all')