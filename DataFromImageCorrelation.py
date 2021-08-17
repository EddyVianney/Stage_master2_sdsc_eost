from TimeSerie import TimeSerie
from Pixel import Pixel
import numpy as np
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from datetime import date, datetime
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_regression
from statsmodels.tsa.stattools import adfuller
from sklearn.manifold import TSNE
import csv
import multiprocessing
import itertools
import operator
import math
import seaborn as sns
plt.style.use('fivethirtyeight')
from osgeo import gdal
from subprocess import Popen
import simplekml
import copy
from sklearn.cluster import DBSCAN
import numpy.ma as ma
from numpy import linalg as LA

class DataFromImageCorrelation():

    def __init__(self, pixels):
        self.pixels = pixels
        self.n_cores = multiprocessing.cpu_count()
        self.sigma = self.compute_vlm_std()
        self.non_filtered_ids = []
        self.velocities = []

    def compute_vel(self, ns_component, ew_component):
        return math.sqrt(ns_component * ns_component + ew_component * ew_component)

    def filter_by(self, factor=1.5, azimuth_tol=30, azimuth_pr=0.6, min_slope=15, alpha=0.05):
        self.non_filtered_ids.clear()
        vfunc = np.vectorize(self.is_to_select)
        x = vfunc(np.arange(len(self.pixels)), factor, azimuth_tol, azimuth_pr, min_slope, alpha)
        self.non_filtered_ids = x.nonzero()[0].tolist()

    def filter_by2(self, factor=1.5, azimuth_tol=30, azimuth_pr=0.6, min_slope=15, alpha=0.05):
        self.non_filtered_ids.clear()
        arguments = [(n, factor, azimuth_tol, azimuth_pr, min_slope, alpha) for n in range(len(self.pixels))]

        results = None
        with multiprocessing.Pool(self.n_cores) as p:
            results = p.starmap(self.is_to_select, arguments)

        self.non_filtered_ids = results.tolist()

    def is_to_select(self, n, factor, azimuth_tol, azimuth_pr, min_slope, alpha):
        is_random = (self.pixels[n].ns.kurtosis().values[0] < 0) and (self.pixels[n].ew.kurtosis().values[0] < 0)
        is_moving = self.pixels[n].get_mean_velocity() > factor * self.sigma
        is_steep = self.pixels[n].is_steep(min_slope)
        is_azimuth_constant = self.pixels[n].is_azimuth_constant(azimuth_tol, azimuth_pr)
        is_linear_regression_significant = (self.pixels[n].is_ns_linear_regression_significant(alpha) and
                                            self.pixels[n].is_ew_linear_regression_significant(alpha))
        return (is_moving and
                is_steep and
                not is_random and
                is_azimuth_constant and
                is_linear_regression_significant)

    def debug_filter_by(self, factor=1):
        vfunc = np.vectorize(self.debug_is_to_select)
        return vfunc(np.arange(len(self.pixels)), factor)

    def compute_vlm_std(self):
        return np.std(
            np.array([math.sqrt(pixel.ns_vel * pixel.ns_vel + pixel.ew_vel * pixel.ew_vel) for pixel in self.pixels]))

    def compute_inst_vels(self, series):
        with multiprocessing.Pool(self.n_cores) as p:
            results = p.map(TimeSerie().compute_inst_vel, series)
            return results

    def compute_velocities(self):
        self.velocities.clear()
        if len(self.non_filtered_ids) == 0:
            self.non_filtered_ids = np.arange(len(self.pixels))
        ns_vels = self.compute_inst_vels([self.pixels[n].ns for n in self.non_filtered_ids])
        ew_vels = self.compute_inst_vels([self.pixels[n].ew for n in self.non_filtered_ids])
        for i in range(len(ns_vels)):
            vels = []
            column = ns_vels[0].columns[0]
            for ns_vel, ew_vel in zip(ns_vels[i][column], ew_vels[i][column]):
                vels.append(self.compute_vel(ns_vel, ew_vel))
            df = pd.DataFrame(vels, index=ns_vels[0].index, columns=['magnitude'])
            self.velocities.append(df)

    def set_slope_map_path(self, slope_map_path):
        self.slope_map_path = slope_map_path

    # TODO : rajouter une erreur en cas de fichier non existant
    def load_raster(self, raster_path):
        return gdal.Open(raster_path)

    def compute_dem_attribute(self, dem_path, dem_attribute_path, attribute):
        if os.path.isfile(dem_path):
            dem = gdal.Open(dem_path)
            dem_attribute = gdal.DEMProcessing(dem_attribute_path, dem, attribute, computeEdges=True)
            dem = None
            dem_attribute = None
        else:
            raise ValueError('Fichier non existant !')