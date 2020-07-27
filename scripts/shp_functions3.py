# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:33:14 2017

@author: shooper
"""

import os
import sys
import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr


def attributes_to_df(shp):
    ''' 
    Copy the attributes of a shapefile to a pandas DataFrame
    
    Parameters:
    shp -- path to a shapefile
    
    Returns:
    df -- a pandas DataFrame. FID is the index.
    '''
    ds = ogr.Open(shp)
    lyr = ds.GetLayer()
    lyr_def = lyr.GetLayerDefn()
    
    fields = [lyr_def.GetFieldDefn(i).GetName() for i in range(lyr_def.GetFieldCount())]
    
    if lyr.GetFeatureCount() == 0:
        raise RuntimeError('Vector dataset has 0 features: ', shp)
    
    vals = []
    for feature in lyr:
        #feature = lyr.GetFeature(i)
        these_vals = {f: feature.GetField(f) for f in fields}
        these_vals['fid'] = feature.GetFID()
        vals.append(these_vals)
        feature.Destroy()
        
    df = pd.DataFrame(vals)
    df.set_index('fid', inplace=True)
    
    return df


def df_to_shp(dataframe, in_shp, out_shp, copy_fields=True, df_id_field=None, shp_id_field=None):
    '''
    Write a new shapefile with features from in_shp and attributes from df
    
    Parameters:
    dataframe -- dataframe containing info to append to the attribute table of the 
          output shapefile
    in_shp -- input shapefile
    out_shp -- path of the output shapefile with extenion ".shp"
    copy_fields -- If True, all fields from in_shp will be copied to the
            attribute table of the output shapefile
    
    Returns: nothing
    
    '''
    df = dataframe.copy()
    
    if 'fid' not in [c.lower() for c in df.columns]:
        print ('Warning: no FID column found in dataframe. Using index of'+\
        ' dataframe instead')
        df['FID'] = df.index
    df.set_index('FID', inplace=True)

    # Get info from ds_in
    ds_in = ogr.Open(in_shp)
    lyr_in = ds_in.GetLayer()
    srs = lyr_in.GetSpatialRef()
    lyr_in_def = lyr_in.GetLayerDefn()
    

    # Make new shapefile and datasource, and then a layer from the datasource
    #driver = ogr.GetDriverByName('ESRI Shapefile')
    driver = ds_in.GetDriver()
    try: ds_out = driver.CreateDataSource(out_shp)
    except: print ('Could not create shapefile with out_shp: \n', out_shp)
    lyr = ds_out.CreateLayer(os.path.basename(out_shp)[:-4], srs, geom_type=lyr_in.GetGeomType())
    # Copy the schema of ds_in
    if copy_fields:
        for i in range(lyr_in_def.GetFieldCount()):
            field_def = lyr_in_def.GetFieldDefn(i)
            lyr.CreateField(field_def)
    
    # Add fields for each of the columns of df 
    df.columns = [ c[:10] for c in df.columns]
    
    for c in df.columns:
        dtype = str(df[c].dtype).lower()
        if 'int' in dtype: lyr.CreateField(ogr.FieldDefn(c, ogr.OFTInteger))
        elif 'float' in dtype: lyr.CreateField(ogr.FieldDefn(c, ogr.OFTReal))
        else: # It's a string
            width = df[c].apply(len).max() + 10
            field = ogr.FieldDefn(c, ogr.OFTString)
            field.SetWidth(width)
            lyr.CreateField(field)
    
    lyr_out_def = lyr.GetLayerDefn() # Get the layer def with all the new fields
    for fid, row in df.iterrows():
        # Get the input feature and create the output feature
        if not (df_id_field and shp_id_field):
            feat_in = lyr_in.GetFeature(fid)
        else:
            for i in xrange(lyr_in.GetFeatureCount()):
                feature = lyr_in.GetFeature(i)
                if feature.GetField(shp_id_field) == row[df_id_field]:
                    feat_in = feature#'''
                    break
        #feat_in = lyr_in.GetFeature(fid)
        feat_out = ogr.Feature(lyr_out_def)
        feat_out.SetFID(fid)
        
        [feat_out.SetField(name, val) for name, val in row.iteritems()]
        if copy_fields:
            [feat_out.SetField(lyr_in_def.GetFieldDefn(i).GetName(), feat_in.GetField(i)) for i in range(lyr_in_def.GetFieldCount())]
        geom = feat_in.GetGeometryRef()
        feat_out.SetGeometry(geom.Clone())
        lyr.CreateFeature(feat_out)
        feat_out.Destroy()
        feat_in.Destroy()

        
    ds_out.Destroy()
    '''Maybe check that all fids in lyr_in were used'''
    
    # Write a .prj file so ArcGIS doesn't complain when you load the shapefile
    srs.MorphToESRI()
    prj_file = out_shp.replace('.shp', '.prj')
    with open(prj_file, 'w') as prj:
        prj.write(srs.ExportToWkt()) 
    
    print ('Shapefile written to: \n', out_shp)