# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:24:07 2017

@author: shooper
"""
import sys
import inspect
import numpy as np
from scipy import stats


def mode(ar, axis=None, nodata=None):
    ''' 
    Code from internet to get mode along given axis faster than stats.mode()
    '''
    if ar.size == 1:
        return (ar[0],1)
    elif ar.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ar.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ar.ndim))

    srt = np.sort(ar, axis=axis)
    dif = np.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    
    return modals#, counts

def lc_proportion(ar): # added by PeterC 11072019 as a place holder for landcover preportion
    #we get an array for each hexagon as 'ar'
    totalpixels = ar.size
    land = np.count_nonzero(ar == 1)
    lc_number = float(land)/float(totalpixels)*100
    # needs to return a single value. to simple but hack like hard code the landcover value here
    return lc_number

def mode_alt(ar):  # added by PeterC 11062019
    return stats.mode(ar)


def npsum(ar, axis=None):
    return np.sum(ar, axis=axis)


def nansum(ar, axis=None):
    return np.nansum(ar, axis=axis)


def mean(ar, axis=None):
    return np.mean(ar, axis=axis)


def nanmean(ar, axis=None):
    
    return np.nanmean(ar, axis=axis)


def count(ar):
    
    return ar.size


def rmse(x, y):
    #import pdb; pdb.set_trace()
    rmse = np.sqrt((((x - y)) ** 2).mean())
    
    return rmse
    

def rmspe(x, y):
    ''' Return root mean square percentage error of y with respect to x'''
    rmspe = np.sqrt(((100 * (x - y)/x) ** 2).mean())
    
    return rmspe


def agree_coef(x, y):
    ''' 
    Return the agreement coefficient, the systematic agreement, and
    unsystematic agreement
    '''
    mean_x = x.mean()
    mean_y = y.mean()
    
    # Calc agreement coefficient (ac)
    ssd  = np.sum((x - y) ** 2)
    dif_mean = abs(mean_x - mean_y)
    spod = np.sum((dif_mean + np.abs(x - mean_x)) * (dif_mean + np.abs(y - mean_y)))
    ac = 1 - (ssd/spod)
    
    # Calc GMFR regression
    r , _ = stats.pearsonr(x, y)
    b = (r/abs(r)) * (np.sum((y - mean_y)**2)/np.sum((x - mean_x)**2)) ** (1./2)
    a = mean_y - b * mean_x
    d = 1./b
    c = -float(a)/b
    gmfr_y = a + b * x
    gmfr_x = c + d * y
    
    # Calc systematic and unsystematic sum of product-difference
    spd_u = np.sum(np.abs(x - gmfr_x) * (np.abs(y - gmfr_y)))
    spd_s = ssd - spd_u
    
    # Calc systematic and unsystematic AC
    ac_u = 1 - (spd_u/spod)
    ac_s = 1 - (spd_s/spod)
    
    return ac, ac_s, ac_u, ssd, spod
    

def pct_nonzero(ar):
    
    mask = ~np.isnan(ar)
    ar = ar[mask]
    n_pixels = float(ar.size)
    if n_pixels == 0:
        return 0
    pct = np.count_nonzero(ar)/n_pixels * 100
    
    return pct
    

def pct_equal_to(ar, val, axis=None):
    
    #add multidimensional support
    #n_equal = np.sum(ar == val, axis=axis)
    n_equal = ar[ar == val].size
    n_total = float(ar.size) #need to figure out total for multi-dim and axis=None
    try:
        pct_equal = n_equal/n_total * 100
    except BaseException:
        pct_equal = -1
        
    return pct_equal


def is_equal_to(ar, center_idx):
    
    center_val = ar[center_idx]
    ar = ar[~np.isnan(ar)]
    if np.all(ar == center_val):
        return 1
    else:
        return 0


def get_function(function_name):
    
    f_dict = {k:obj for k, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj) and k != 'get_function'}
    functions = f_dict.keys()
    if function_name not in functions:
        joined = '\n\t'.join(functions)
        msg = 'Function name not recognized: %s\n\nAvailable functions:\n\t%s' % (function_name, joined)
        raise NameError(msg)
    
    return f_dict[function_name]
