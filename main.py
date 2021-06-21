#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:37:28 2021

@author: eost-user
"""

import csv
import pandas as pd
from Pixel import Pixel
from DataFromImageCorrelation import DataFromImageCorrelation
from Clustering import Clustering
from Visualization import Visualization

def load_data(filename):

    # numéro de la ligne ou commence les données
    num_start = 44
    # numéro de la ligne ou se trouve la liste des dates
    num_list_dates = 40
    # attributs présent dans les données
    columns = ['id', 'Lat','Lon', 'Topo', 'Vel', 'Coer',' CosN', 'CosE', 'CosU']
    # dictionnaire stockant les données
    data = {column: [] for column in columns}
    # liste des dates 
    indexes = []
    # series temporelles
    series = []
    # liste de dataframes
    df_series = []

    with open(DATA_PATH + '/' + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 1 
        for row in csv_reader:
            if line_count == num_list_dates:
                indexes = [row[0].split(' ')[1]] + row[1:]
            if line_count >= num_start:
                # extraction des premiers attributs
                for i in range(len(columns)):
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


if __name__ == '__main__':
    
    DATA_PATH = './donnees'
    # chargement de la composante  Est-Ouest du mouvement du sol
    df_ew, df_ew_ts = load_data('MM_TIO_EW_31TGK_20151227_to_20200906.csv')
    # chargement de la composante Nord-Sud du mouvement du sol
    df_ns, df_ns_ts = load_data('MM_TIO_NS_31TGK_20151227_to_20200906.csv')
    
    # fusionner les fichiers sur la topo et les vitesses (pour éviter la redondance de l'information)
    df_ew.rename(columns={'Vel': 'Vel_ew'}, inplace=True)
    df_ns.rename(columns={'Vel': 'Vel_ns'}, inplace=True)
    geo = pd.concat([df_ew[['id', 'Lat','Lon','Topo','Vel_ew']], df_ns[['Vel_ns']]], axis=1)
    
    n = 2000
    RASTER_FOLDER_PATH = 'rasters'
    DEM_FILENAME =  '31TGK_copernicus_dem.tif'
    Data = DataFromImageCorrelation(RASTER_FOLDER_PATH, DEM_FILENAME, geo.head(n), df_ns_ts[:n], df_ew_ts[:n])
    
    # calcul des profils de vitesse
    Data.compute_velocities(min_slope=5, ampl=2)