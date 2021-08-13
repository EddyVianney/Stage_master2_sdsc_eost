from TimeSerie import TimeSerie
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


class Pixel(TimeSerie):

    def __init__(self, id_, lat, lon, topo, ns, ew, ns_vel, ew_vel, mnt_slope, mnt_azimuth):
        super(TimeSerie, self).__init__()
        self.id = id_
        self.lat = lat
        self.lon = lon
        self.topo = topo
        self.ns = self.impute(ns)
        self.ew = self.impute(ew)
        self.azimuths = self.compute_azimuth_serie()
        self.mnt_slope = mnt_slope
        self.mnt_azimuth = mnt_azimuth
        self.ns_vel = ns_vel
        self.ew_vel = ew_vel
        self.is_selected = False

    def compute_azimuth_serie(self):
        column = self.ns.columns[0]
        ns = self.ns[column].values
        ew = self.ew[column].values
        azimuths = np.arctan2(ew, ns)
        azimuths = np.where(azimuths >= 0, azimuths, azimuths + 2 * np.pi)
        azimuths = np.degrees(azimuths)
        return pd.DataFrame({'azimuth': pd.Series(azimuths, index=self.ns.index)})

    def compute_magnitude_displacement(self):
        x = self.ns.iloc[-1]
        y = self.ew.iloc[-1]
        return math.sqrt(x * x + y * y)

    def compute_vector_coherence(self):
        diff_ns = np.diff(np.squeeze(self.ns.values))
        diff_ew = np.diff(np.squeeze(self.ew.values))
        num = self.compute_norm(np.sum(diff_ns), np.sum(diff_ew))
        den = np.sum(self.compute_norm(diff_ns, diff_ew))
        return num / den

    def compute_norm(self, x, y):
        return np.sqrt(x * x + y * y)

    def has_enough_values(self, pc):
        return self.compute_null_val_percentage(self.ns) < pc

    def is_linear_regression_significant(self, alpha, ampl):
        ns_sm = self.smooth(self.ns, ampl)
        ew_sm = self.smooth(self.ew, ampl)
        return self.compute_linear_reg_pval(ns_sm) < alpha or self.compute_linear_reg_pval(ew_sm) < alpha

    def is_ns_linear_regression_significant(self, alpha):
        return self.compute_linear_reg_pval(self.ns) < alpha

    def is_ew_linear_regression_significant(self, alpha):
        return self.compute_linear_reg_pval(self.ew) < alpha

    def is_linear_regression_significant_(self, alpha):
        return self.is_ns_linear_regression_significant(alpha) or self.is_ew_linear_regression_significant(alpha)

    def compute_vel(self, ns_component, ew_component):
        return math.sqrt(ns_component * ns_component + ew_component * ew_component)

    def is_moving(self, std, factor):
        return self.get_mean_velocity() > factor * std

    def is_to_select1(self, filename, ref, alpha, min_slope, std, factor, pc, ampl_disp):
        return self.is_linear_regression_significant(alpha, ampl_disp) and self.is_moving(std,
                                                                                          factor) and self.is_steep(
            filename, ref, min_slope)

    def is_to_select2(self, filename, ref, min_slope, vel_std, amplitude):
        return self.is_moving(vel_std, amplitude) and self.is_steep(filename, ref, min_slope)

    def compute_slope(self, ref, file):
        val = os.popen('gdallocationinfo -valonly -%s %s %f %f' % (ref, file, self.lat, self.lon)).read()
        if len(val) == 0:
            raise ValueError('La pente est non valide !')
        return float(val)

    def get_mean_velocity(self):
        return math.sqrt(self.ns_vel * self.ns_vel + self.ew_vel * self.ew_vel)

    def smooth_by_iqr(self, serie):
        column = serie.columns[0]
        low, up = self.compute_interquartile_range(serie)
        for n in range(len(serie)):
            if serie.values[n][0] < low or serie.values[n][0] > up:
                serie.iloc[i, serie.columns.get_loc('displacement')] = np.nan
        return serie.interpolate(limit_direction='both', inplace=False)

    def is_sign_random(self):
        ns_prop = self.compute_sign_change_proportion_ns()
        ew_prop = self.compute_sign_change_proportion_ew()
        return abs(ns_prop[0] - ns_prop[1]) > 0.5 or abs(ew_prop[0] - ew_prop[1]) > 0.5

    # calculer la proportion des pixels tels que |mnt_azimuth - azimuth| < tol
    def is_azimuth_constant(self, tol, pr):
        attribute = self.azimuths.columns[0]
        mnt_azimuth = self.mnt_azimuth
        azimuths = self.azimuths[attribute].values
        proportion = np.where((azimuths < mnt_azimuth + tol) & (azimuths > mnt_azimuth - tol))[0].shape[0] / azimuths.shape[0]
        return proportion > pr

    def is_steep(self, min_slope):
        return self.mnt_slope > min_slope
