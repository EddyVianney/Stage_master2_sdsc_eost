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
from osgeo import gdal
from subprocess import Popen
import simplekml
import copy
import numpy.ma as ma
from numpy import linalg as LA
import rasterio
from pyproj import Proj

class Data():
    """
        Attributes
        ----------
        slope_map_raster : raster
                            fichier raster des pentes calculées par gdal à partir d'un Modèle Numérique de Terrain
        azimuth_map_raster :
        slopes_map_values :
        azimuth_map_values:
        epsg_code :
        proj :
        slopes :
        azimuths :
        pixels :
    """

    # numéro de la ligne ou commence les données
    num_start = 44

    # numéro de la ligne ou se trouve la liste des dates
    num_list_dates = 40

    # attributs présent dans les données
    columns = ['id', 'Lat' ,'Lon', 'Topo', 'Vel', 'Coer' ,' CosN', 'CosE', 'CosU']

    # raise en error when the two file have different sizes

    def __init__(self, ew_file_path, ns_file_path, slope_map_path, azimuth_map_path, pc=0.4):
        self.slope_map_raster = rasterio.open(slope_map_path)
        self.azimuth_map_raster = rasterio.open(azimuth_map_path)
        self.slope_map_values = rasterio.open(slope_map_path).read(1)
        self.azimuth_map_values = rasterio.open(azimuth_map_path).read(1)
        self.epsg_code = self.azimuth_map_raster.crs.to_string().lower()
        self.proj = Proj(self.epsg_code, preserve_units=False)
        slopes, azimuths, pixels = self.load(ew_file_path, ns_file_path, pc)
        self.slopes = slopes
        self.azimuths = azimuths
        self.pixels = pixels
        self.mask = None

    def to_date(self, str):
        str_strp = str.strip()
        year, month, day = int(str_strp[0:4]), int(str_strp[4:6]), int(str_strp[6:8])
        return date(year, month, day)

    def load_data_from_csv(self, data_path):
        """
        fonction qui charge les séries temporelles de déplacement et les autres sorties de MPIC(vitesses moyennes, location géopgraphique des pixels, elevation topopgrahique)
        parameters
        ----------
        data_path : str
            chemin des données

        Returns
        -------
            dataframe, dataframe
        """
        columns = Data.columns

        # dictionnaire stockant les données
        data = {column: [] for column in columns}
        # liste des dates
        indexes = []
        # series temporelles
        series = []
        # liste de dataframes
        df_series = []

        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 1
            for row in csv_reader:
                if line_count == Data.num_list_dates:
                    indexes = [row[0].split(' ')[1]] + row[1:]
                if line_count >= Data.num_start:
                    # extraction des premiers attributs
                    for i in range(len(Data.columns)):
                        data[columns[i]].append(row[i])
                    # extraction de l'attribut TS(série temporelle)
                    series.append([float(v) for v in row[len(columns):]])
                line_count  += 1
            if len(indexes) != len(series[0]):
                print('Erreur : Les indexes et les valeurs ne correspondent pas')
            # convertir les index en date
            indexes = [d.strip()[0:8] for d in indexes]
            # créer une liste de dataframes, chacun contenant une série temporelle
            for serie in series:
                tmp_serie = pd.DataFrame({'displacement': pd.Series(serie, index=pd.DatetimeIndex(indexes))})
                tmp_serie.sort_index(inplace=True)
                df_series.append(tmp_serie)
            # creer un dataframe pour les autres attributs
            df = pd.DataFrame(data)
            for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            df.set_index('id')

        return df, df_series

    def set_pixels(self, pixels):
        self.pixels = pixels

    def set_slopes(self, slopes):
        self.slopes = slopes

    def set_azimuths(self, azimuths):
        self.azimuths = azimuths

    def load(self, ew_file_path, ns_file_path, pc=0.4):
        """

        parameters
        ----------
        ew_file_path : str
            chemin du fichier csv contenant les sorties MPIC des déplacements Est-Ouest
        ns_file_path : str
            chemin du fichier csv contenant les sorties MPIC des déplacements Nord-Sud
        pc :  float
            proportion maximale de valeurs non nulles que ne doit pas dépasser une série temporelle pour être considérée comme exploitable
        Returns
        -------
        """
        slopes, azimuths = None, None

        df_ew, ew = self.load_data_from_csv(ew_file_path)
        df_ns, ns = self.load_data_from_csv(ns_file_path)

        df_ew.rename(columns={'Vel': 'Vel_ew'}, inplace=True)
        df_ns.rename(columns={'Vel': 'Vel_ns'}, inplace=True)
        geo = pd.concat([df_ew[['id', 'Lat' ,'Lon' ,'Topo' ,'Vel_ew']], df_ns[['Vel_ns']]], axis=1)

        pixels = []
        n = len(ns[0])

        for i in range(len(ns)):
            if ns[i].isnull().sum().sum() / n < pc:
                row, col = self.get_position_in_raster(geo.iloc[i].Lat, geo.iloc[i].Lon)
                mnt_slope = self.get_value_from_slope_map2(row, col)
                mnt_azimuth = self.get_value_from_azimuth_map2(row, col)
                pixels.append(Pixel(geo.iloc[i].id, geo.iloc[i].Lat, geo.iloc[i].Lon, geo.iloc[i].Topo, ns[i], ew[i], geo.iloc[i].Vel_ns, geo.iloc[i].Vel_ew, mnt_slope, mnt_azimuth))

        return slopes, azimuths, pixels


    def convert_lat_lon_to_utm(self, lat, lon):
        """
        fonction qui convertit un les coordonnées lat, lon en coordonnées UTM
        parameters
        ----------
        lat : float
            latitude
        lon : float
            longitude
        Returns
        -------
            tuple
        """
        return self.proj(lat, lon)

    def get_position_in_raster(self, lat, lon):
        """
        fonction qui permet d'avoir la position (en termes d'indice) d'un pixel à partir de ses coordonnées géographiques
        parameters
        ----------
        lat :foat
            latitude
        lon: float
            longitude
        Returns
        -------
            int, int
                indices de ligne et de colonne
        """
        x, y = self.convert_lat_lon_to_utm(lat, lon)
        row, col = self.azimuth_map_raster.index(x, y)
        return row, col

    def get_value_from_slope_map2(self, row, col):
        """
        fonction qui permet d'extraire la valeur de la pente topographique à partir de la position (en termes d'indices) d'un pixel
        parameters
        ----------
        row : int
            indice de ligne
        col : int
            indice de colonne
        Returns
        -------
            float
                valeur de la pente située à la position [row, col]
        """
        return self.slope_map_values[row, col]

    def get_value_from_azimuth_map2(self, row, col):
        """
        fonction qui permet d'extraire la valeur de l'azimut à partir de la position (en termes d'indices) d'un pixel
        parameters
        ----------
        row : int
            indice de ligne
        col : int
            indice de colonne
        Returns
        -------
            float
                valeur de l'azimut situé à la position [row, col]
        """
        return self.azimuth_map_values[row, col]

    def get_value_from_slope_map(self, lat, lon):
        x, y = self.convert_lat_lon_to_utm(lat, lon)
        row, col = self.slope_map_raster.index(x, y)
        return self.slope_map_values[row, col]

    def get_value_from_azimuth_map(self, lat, lon):
        x, y = self.convert_lat_lon_to_utm(lat, lon)
        row, col = self.azimuth_map_raster.index(x, y)
        return self.azimuth_map_values[row, col]

    def extract_value_from_raster(self, lat, lon, file, ref='wgs84'):
        """
        fonction permettant d'extraire une valeur d'un fichier raster
        parameters
        ----------
        lat : float
            latitude
        lon : float
            longitude
        file : str
            chemin du fichier raster
        ref : str
            système de référence
        Returns
        -------
            float
                valeur du raster à la position spécifiée
        """
        val = os.popen('gdallocationinfo -valonly -%s %s %f %f' % (ref, file, lat, lon)).read()
        if len(val.strip()) == 0:
            return -9999
        else:
            return float(val)