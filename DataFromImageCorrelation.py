#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:27:02 2021

@author: PAMBOU MOUBOGHA Eddy Vianney
"""
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from statsmodels.tsa.stattools import adfuller
from sklearn.manifold import TSNE
import csv
import multiprocessing
import itertools
import operator
import math
import json
from osgeo import gdal 
from subprocess import Popen
import simplekml
from TimeSerie import TimeSerie

class DataFromImageCorrelation():
    
    def __init__(self, raster_folder_name, dem_filename, geo, ns, ew, pc=0.4, alpha=0.05, ref='wgs84'):
        self.ns, self.ew  = self.load(geo, ns, ew, pc)
        self.compute_slope_map(raster_folder_name, dem_filename)
        self.n_cores = multiprocessing.cpu_count()
        self.non_filtered_ids = []
        self.magnitudes = None
        self.pc = pc 
        self.alpha = alpha
        self.ref = ref
    
    def set_slope_map_path(self, slope_map_path):
        self.slope_map_path = slope_map_path
        
    def select_pixel(self, n, min_slope):
        return (self.ns[n].is_to_select(self.slope_map_path, self.ref, min_slope, self.alpha) or 
                self.ew[n].is_to_select(self.slope_map_path, self.ref, min_slope, self.alpha))
    
    def load_raster(self, raster_folder_name, raster_filename):
        return gdal.Open(raster_folder_name + '/' +  raster_filename)
        
    def compute_slope_map(self, raster_folder_name, dem_name):
        dem = None
        slope_map = None
        slope_map_name = dem_name.split('.')[0] + '_' + 'slope_map.tif'
        slope_map_path = raster_folder_name + '/' + slope_map_name
        if not os.path.isfile(slope_map_path):
            dem = gdal.Open(raster_folder_name + '/'+ dem_name)
            slope_map = gdal.DEMProcessing(slope_map_path, dem, 'slope', computeEdges = True)
        self.set_slope_map_path(slope_map_path)
    
    def empty_non_filtered_ids(self):
        self.non_filtered_ids.clear()
        
    def add_non_filtered_ids(self, n):
        self.non_filtered_ids.append(n)
    
    def load(self, geo, ns, ew, pc):
        ns_r, ew_r = [], []
        for i in range(len(ns)):
            # vérifier que les séries temporelles comporte un pourcentage de valeurs null inférieure à pc et on des id identiques
            if ns[i].isnull().sum().sum()/len(ns[i]) < pc:
                ns_ts = ns[i].interpolate(limit_direction='both', inplace=False)
                ew_ts = ew[i].interpolate(limit_direction='both', inplace=False)   
                ns_r.append(TimeSerie(geo.iloc[i].id, geo.iloc[i].Lat, geo.iloc[i].Lon, geo.iloc[i].Topo, ns_ts))
                ew_r.append(TimeSerie(geo.iloc[i].id, geo.iloc[i].Lat, geo.iloc[i].Lon, geo.iloc[i].Topo, ew_ts))
        return ns_r, ew_r
    
    def compute_vel(self, ns_component, ew_component):
        return math.sqrt(ns_component * ns_component + ew_component * ew_component)
    
    def compute_magnitude(self, boolean, min_slope=None):
        magnitudes = []
        ns_vel_ts = None
        ew_vel_ts = None
        if boolean:
            ns_vel_ts = self.compute_inst_vels(self.ns)
            ew_vel_ts = self.compute_inst_vels(self.ew)
        else:
            ns, ew    = self.transform(min_slope)
            ns_vel_ts = self.compute_inst_vels(ns)
            ew_vel_ts = self.compute_inst_vels(ew)
        for i in range(len(ns_vel_ts)):
            vels = []
            column = ns_vel_ts[0].columns[0]
            for ns_component, ew_component in zip(ns_vel_ts[i][column], ew_vel_ts[i][column]):
                vels.append(self.compute_vel(ns_component, ew_component))
            df = pd.DataFrame(vels, index=ns_vel_ts[0].index, columns=['magnitude'])
            magnitudes.append(df)
        return magnitudes
            
    def compute_inst_vel(self, ts):
        vels =  []
        for i in range(1, len(ts.serie)-1):
            duration = (ts.serie.index[i+1] - ts.serie.index[i-1]).days
            displacement = ts.serie.iloc[i+1].values[0] - ts.serie.iloc[i-1].values[0]
            vels.append(displacement / duration)
        # supprimer le premier et le dernier index (formule non applicable)
        return pd.DataFrame(vels, index=ts.serie.index[1:-1], columns=['vel'])
    
    def compute_inst_vels(self, ts):
        with multiprocessing.Pool(self.n_cores) as p:
            results = p.map(self.compute_inst_vel, ts)
            return results
        
    # attention pas de normalisation dans cette méthode (car elle est utilise pour le calul de vitesse)
    def transform(self, min_slope):
        ns, ew = [], []
        self.empty_non_filtered_ids()
        for n in range(len(self.ns)):
            if self.select_pixel(n, min_slope):
                ns.append(self.ns[n].smooth().detect_non_moving_serie(self.alpha))
                ew.append(self.ew[n].smooth().detect_non_moving_serie(self.alpha))
                self.add_non_filtered_ids(n)
        return ns, ew
    
    # renvoyer les données pour effectuer le clustering
    # si on n'interesse qu'à la forme, il faut normaliser les données
    def prepare(self, min_slope):
        self.set_magnitudes(self.compute_magnitude(False, min_slope))
    
    # for the moment, we only normalize velocity time series
    def normalize(self):
        return np.array([StandardScaler().fit_transform(df).reshape(len(df)) for df in self.magnitudes])
    
    def set_magnitudes(self, magnitudes):
        self.magnitudes = magnitudes
