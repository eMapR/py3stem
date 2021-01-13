import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import json 
import os
import glob
import numpy as np 
from osgeo import gdal
from PIL import Image
from matplotlib import colors


# with open("/vol/v3/ben_ak/vector_files/animations/sheriden_glacier_revised.geojson") as f:
# 	geoms = json.load(f)	
# 	print(geoms)


# load the raster, mask it by the polygon and crop it
def make_aoi_rasters(aoi,raster,output_directory): 
	with rasterio.open(raster) as src:
		output_filename = output_directory+os.path.split(raster)[1][:-4]+'_sheriden_glacier_revised.tif'
		out_image, out_transform = mask(src, aoi, crop=True)
		out_meta = src.meta.copy()

	#save the resulting raster  
	out_meta.update({"driver": "GTiff",
	"height": out_image.shape[1],
	"width": out_image.shape[2],
	"transform": out_transform})

	with rasterio.open(output_filename, "w", **out_meta) as dest:
		dest.write(out_image)

def make_png_files(raster,output_directory,year): 
	# start the for loop to create one map per year
	#for year in list_of_years:
	#with rasterio.open(raster) as src: 
	#image = src.read(1)#.astype('float')
	# create map, UDPATE: added plt.Normalize to keep the legend range the same for all maps
	ds = gdal.Open(raster).ReadAsArray()
	ds[ds==12]=1
	ds[ds!=1] = 0 
	fig = plt.figure()#figsize=(16,8))
	ax = fig.add_subplot(111)
	#cmap=['#000000','#ffffff']
	# make a color map of fixed colors
	cmap = colors.ListedColormap(['#a6a6a6', '#0c2b66'])
	bounds=[0,0.5,1]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	ax.imshow(ds,cmap=cmap,norm=norm,interpolation='nearest')#,origin='lower')
	#plt.show()
	ax.annotate(year,xy=(0.28, .2), xycoords='figure fraction',
	        horizontalalignment='left', verticalalignment='top',
	        fontsize=28)
	ax.axis('off')
	
	  # this will save the figure as a high-res png in the output path. you can also save as svg if you prefer.
	filepath = os.path.join(output_directory, year+'_sheriden_glacier_revised.jpg')
	plt.savefig(filepath, dpi=300)
	# plt.show()
	# plt.close('all')
	


def make_gif(image_directory): 

	# filepaths
	fp_in = image_directory+"*revised.jpg"
	fp_out = image_directory+"sheriden_glacier_revised_1000.gif"

	# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
	img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
	img.save(fp=fp_out, format='GIF', append_images=imgs,
	         save_all=True, duration=1000, loop=0)
def main(): 
	output_directory = '/vol/v3/ben_ak/raster_files/animations/'
	#depreceated
	# geoms = [{'type': 'Polygon', 'coordinates': [[(468005.66125929693, 1207106.0079712374), (476981.05697424227, 1218067.1107027635), (491715.0030372721, 1214770.8370552394), (482382.1800593421, 1204723.1595513404), 
	# (478410.76602618047, 1200712.031377847), (471500.5056084792, 1197971.7556949656), (468799.9440659293, 1202737.4525347594), (468005.66125929693, 1207106.0079712374)]]}]

	geoms=[{'type': 'Polygon', 'coordinates': [[(468760.22992559744, 1200394.318255194), (472334.502555443, 1211593.7058287098), (479562.4760957972, 1209568.2846717974), (476742.7721322524, 1198726.324361266), (468760.22992559744, 1200394.318255194)]]}]


	# for file in glob.glob("/vol/v3/ben_ak/param_files_rgi/southern_region/output_files/*2016_full_run_vote.tif"): 
	# 	print(file)
	# 	make_aoi_rasters(geoms,file,output_directory)

	# for file in sorted(glob.glob("/vol/v3/ben_ak/raster_files/animations/*sheriden_glacier_revised.tif")): 
	# 	make_png_files(file,output_directory,os.path.split(file)[1].split('_')[1])
	
	make_gif('/vol/v3/ben_ak/raster_files/animations/')

if __name__ == '__main__':
	main()

# fig=plt.figure()
	# plt.imshow(ds)
	# #fig = plt.plot(image)#column=year, cmap='Blues', figsize=(10,10), linewidth=0.8, edgecolor='0.8', vmin=vmin, vmax=vmax,
	# #legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	# # remove axis of chart
	#fig.axis('off')
	#plt.show()
	#plt.close()
# # # add a title
	# # # fig.set_title('Violent crimes in London', \
	# # #           fontdict={'fontsize': '25',
	# # #                      'fontweight' : '3'})

	# # create an annotation for the year by grabbing the first 4 digits
	# #only_year = year[:4]
	# # position the annotation to the bottom left
	# fig.annotate(year,
	#         xy=(0.1, .225), xycoords='figure fraction',
	#         horizontalalignment='left', verticalalignment='top',
	#         fontsize=35)

	#   # this will save the figure as a high-res png in the output path. you can also save as svg if you prefer.
	# filepath = os.path.join(output_directory, year+'_test.jpg')
	# # chart = fig.get_figure()
	# # chart.savefig(filepath, dpi=300)
	# with rasterio.open(raster) as src:
	# 	arr = src.read(1)
	# 	arr = np.where(arr==12,1,0)
	# 	meta = src.meta
	# 	meta(
	# 	dtype=rasterio.uint8,
	# 	nodata=0,
	# 	count=3)

	# with rasterio.open(filepath, 'w', **meta) as dst:
	# 	dst.write_band(1, r.astype(rasterio.uint8))
		# dst.write_band(2, g.astype(rasterio.uint8))
		# dst.write_band(3, b.astype(rasterio.uint8))