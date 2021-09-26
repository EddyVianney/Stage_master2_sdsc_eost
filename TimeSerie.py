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
from scipy.spatial.distance import euclidean
from sklearn.feature_selection import f_regression
from statsmodels.tsa.stattools import adfuller
from sklearn.manifold import TSNE
from DBCV import DBCV
import hdbscan
import csv
import multiprocessing
import itertools
import operator
import math
import seaborn as sns
import json
plt.style.use('fivethirtyeight')
from osgeo import gdal
from subprocess import Popen
import simplekml
import copy
from sklearn.cluster import DBSCAN
import numpy.ma as ma
from pyproj import Proj


class TimeSerie():

    def __init__(self):
        pass

    def has_null_values(self, serie):
        return self.count_null_values(serie) > 0

    def count_null_values(self, serie):
        return serie.isnull().sum().sum()

    def compute_null_val_percentage(self, serie):
        return 100 * (1.0 * self.count_null_values(serie) / len(serie))

    def interpolate(self, serie):
        return serie.interpolate(limit_direction='both', inplace=False)

    def compute_pearson_coef(self, serie):
        return stats.pearsonr(np.squeeze(serie.values), get_days(serie.index))

    def compute_linear_reg_pval(self, serie):
        # extraire X et y
        X, y = self.prepare(serie)
        # calculer la p-value de la regression lineaire
        _, pval = f_regression(X, y.ravel())
        return pval[0]

    def select(self, serie, filename, ref, min_slope, alpha, sigma, ampl, pc):
        slope = self.get_slope_value(ref, serie, filename)
        p_value = self.get_linear_reg_pval(serie, alpha)
        vlm = self.vlm
        # filtrage des series avec peu de valeurs
        if self.compute_nul_val_percentage > pc:
            return False
        # filtrage des regressions non significatives
        if p_value > alpha:
            return False
        # filtrage des vitesses faibles
        if abs(vlm) < ampl * sigma:
            return False
        # filtrage des pentes faibles
        if slope < min_slope:
            return False
        # sauvegarder l'état du pixel
        self.set_selected()
        return True

    def normalize(self):
        return StandardScaler().fit_transform(self.serie)

    # la copie renvoie bien un nouvel objet, il n'y a pas d'effets de bord
    def smooth(self, s, ampl):
        serie = s.copy()
        std = math.sqrt(serie.var())
        for i in range(len(serie)):
            if abs(serie.iloc[i].displacement) > ampl * std:
                serie.iloc[i, serie.columns.get_loc('displacement')] = np.nan
        return serie.interpolate(limit_direction='both', inplace=False)

    def detect_trend(self, alpha):
        X, y = self.prepare()
        # calculer la p-value de la regression lineaire
        _, pval = f_regression(X, y.ravel())
        self.set_linear_reg_pvalue(pval)
        # eliminer le bruit si la regression n'est pas significative (bourrage de zeros)
        if pval > alpha:
            # return self.clone(pd.DataFrame(0.0, index=self.serie.index, columns=self.serie.columns))
            return self.deepcopy(pd.DataFrame(0.0, index=self.serie.index, columns=self.serie.columns))
        else:
            # return self.clone(self.serie.copy())
            return self.deepcopy(self.serie)

    def transform(self, alpha):
        return smooth().detect_non_moving_serie(alpha)

    def deepcopy(self, serie):
        clone = copy.deepcopy(self)
        clone.set_serie(serie)
        return clone

    def compute_adfuller(self, serie):
        adf_result = adfuller(serie)
        adf_output = pd.Series(adf_result[0:4],
                               index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in adf_result[4].items():
            adf_output['Critical Value (%s)' % (key)] = value
        return adf_output[1]

    # ce test nécessite d'avoir des données regulièrement échantillonnées
    # Hyopthèse nulle : il existe une racine unitaire (série croissante ou cyclique)
    # Hypthèse alternative : il n'existe pas de racine unitaire (série stationnaire)
    # Si la p-valeur du test est inférireure à O.05, on rejette l'hypothèse nulle et la sériee est stationnaire
    # NB: on recherche des signaux non stationnaires
    def is_stationary(self, serie, freq='D', alpha=0.05):
        resampled = serie.resample(freq)
        interpolated = upsampled.interpolate(method='linear')
        return self.compute_adfuller(interpolated) < alpha

    def get_days(self, serie):
        days = []
        dates = serie.index
        for i in range(len(dates)):
            days.append(abs((dates[0] - dates[i]).days))
        return days

    def get_Xy(self, serie):
        X = get_days(serie.index)
        y = StandardScaler().fit_transform(serie)
        return np.array(X).reshape(-1, 1), y

    def prepare(self, serie):
        # transformer les index en durée pour pouvoire effectuer une regression linéaire
        X = np.array([abs((serie.index[0] - serie.index[n]).days) for n in range(len(serie.index))]).reshape(-1, 1)
        # extraire la cible
        y = StandardScaler().fit_transform(serie)
        return X, y

    # that functions gives approximately the same result when use sklearn linear regression
    # x and y and numpy array
    def compute_slope(self, serie):
        x, y = self.prepare(serie)
        return np.cov(x.T, y.T)[0][1] / np.var(x)

    def compute_inst_vel(self, serie):
        vels = []
        for i in range(1, len(serie) - 1):
            duration = (serie.index[i + 1] - serie.index[i - 1]).days
            displacement = serie.iloc[i + 1].values[0] - serie.iloc[i - 1].values[0]
            vels.append(displacement / duration)
        return pd.DataFrame(vels, index=serie.index[1:-1], columns=['vel'])

    def impute(self, serie, ):
        if self.has_null_values(serie):
            return serie.interpolate(limit_direction='both', inplace=False)
        else:
            return serie

    def compute_diff_vect(self, serie):
        disp = np.diff(np.squeeze(serie.values))
        duration = np.diff(np.squeeze(serie.index)) / np.timedelta64(1, 'D')
        return disp, np.cumsum(duration)
