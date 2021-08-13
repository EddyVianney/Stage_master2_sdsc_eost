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
        self.uu = uu
        self.vv = vv

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

    def get_component_filenames(data_path, component):
        filenames = []
        for filename in os.listdir(data_path):
            if 'FILTER' in filename and  component in filename:
                filenames.append(filename)
        return filenames
