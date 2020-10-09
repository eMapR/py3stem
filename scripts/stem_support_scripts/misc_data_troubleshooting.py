"""Generic script container for misc functions for data cleaning, formatting etc."""

import os
import sys
import glob
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt


def assess_date_distribution(csv_list):
	fig,ax = plt.subplots(1,3)
	ax = ax.flatten()
	count = 0
	for file in csv_list:  
		df = pd.read_csv(file,parse_dates=True)
		#print(df.head())
		if '.geo' in df.columns: 
			df.drop(columns=['.geo'])
		df['month'] = pd.DatetimeIndex(df['date']).month

		print(df.head())
		ax[count].hist(df['month'],color = 'darkgreen')
		ax[count].set_xlabel('Month')
		ax[count].set_ylabel('Image count')
		ax[count].set_xticks([7,8,9])

		ax[count].set_title(os.path.split(file)[1][:4]+' composites')
		count += 1
	plt.show()


def main(): 

	year_2001=("/vol/v3/ben_ak/excel_files/2001_ic_dates_full.csv")
	year_2009=("/vol/v3/ben_ak/excel_files/2009_ic_dates_full.csv")
	year_2019=("/vol/v3/ben_ak/excel_files/2019_ic_dates_full.csv")
	years = [year_2001,year_2009,year_2019]
	assess_date_distribution(years)
if __name__ == '__main__':
	main()