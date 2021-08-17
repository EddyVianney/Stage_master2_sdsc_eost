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


class Ica():

    def __init__(self, data_path):
        uu, vv = self.load_displacements(data_path)
        uu_4ica, vv_4ica = self.prepare(uu, vv)
        self.n_filenames = uu.shape[1] + vv.shape[1]
        self.uu = uu
        self.vv = vv
        self.uu_4ica = uu_4ica
        self.vv_4ica = vv_4ica
        self.aa = self.build_aa()

    def load_displacements(self, data_path):
        uu, vv = [], []
        for filename in os.listdir(data_path):
            if 'FILTER' in filename:
                raster = gdal.Open(data_path + filename)
                values = raster.GetRasterBand(1).ReadAsArray()
                if 'MM_NS' in filename:
                    vv.append(values)
                if 'MM_EW' in filename:
                    uu.append(values)
        return np.array(uu), np.array(vv)

    def get_component_filenames(self, data_path, component):
        filenames = []
        for filename in os.listdir(data_path):
            if 'FILTER' in filename and  component in filename:
                filenames.append(filename)
        return filenames

    def build_aa(self):
        a = []
        for i in range(0, self.n_filenames - 1):
            for j in range(i+1, self.n_filenames):
                        b = np.zeros(self.n_filenames - 1)
                        b[i:j] = 1
                        a.append(b)
        return np.linalg.pinv(np.transpose(a).dot(a)).dot(np.transpose(a))

    def prepare(self, uu, vv):
        vv_4ica = []
        uu_4ica = []
        for i in range(0, uu.shape[1]):
            for j in range(0, vv.shape[2]):
                vv_4ica.append(uu[:, i, j])
                uu_4ica.append(vv[:, i, j])
        return uu_4ica, vv_4ica

    def ica(self, n_components, whiten='boolean'):
        icav = FastICA(n_components=n_components, whiten=whiten)
        icau = FastICA(n_components=n_components, whiten=whiten)

        S_v = icav.fit_transform(self.vv_4ica)  # Reconstruct signals
        S_u = icau.fit_transform(self.uu_4ica)  # Reconstruct signals
        S_v_t = np.transpose(S_v)
        S_u_t = np.transpose(S_u)
        A_v = icav.mixing_
        A_u = icau.mixing_

        du = []
        dv = []

        for i in range(0, n_components):
            du.append(self.aa.dot(A_u[:, i]))
            dv.append(self.aa.dot(A_v[:, i]))
        return du, dv

