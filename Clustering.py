#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:26:18 2021

@author: PAMBOU MOUBOGHA Eddy Vianney
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import hdbscan
import itertools
import operator
import math
from dtaidistance import dtw
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.cluster import DBSCAN

class Clustering():
    
    def __init__(self, data):
        self.data = data
        self.result = None
        self.n_clusters = None
        
    def select_scaler(serie, option):
        if option == 0:
            return StandardScaler().fit_transform(serie)
        else:
            return RobustScaler.fit_transform(serie)
        
    def normalize(self, series):
        return np.array([StandardScaler().fit_transform(serie).reshape(len(serie)) for serie in series])
    
    def compute_similarity_matrix(self, series):
        size = len(series)
        distances_matrix = np.zeros(shape=(size, size))
        for n in range(size):
            for m in range(n , size):
                s1 = StandardScaler().fit_transform(self.magnitudes[n])
                s2 = StandardScaler().fit_transform(self.magnitudes[m])
                dist = dtw.distance(s1, s2)
                distances_matrix[n, m] = dist
                distances_matrix[m, m] = dist
        return distances_matrix
    
    def cluster(self, min_cluster_size, precomputed=False):
        data = self.data.velocities
        if not precomputed:
            self.result = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True).fit(self.normalize(data))
        else:
            self.result = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, metric='precomputed').fit(self.compute_similarity_matrix(data))

    def get_n_clusters(self):
        return 0 if self.result.labels_.max() < 0 else self.result.labels_.max() + 1
    
    def visualize(self):
        projection = TSNE().fit_transform(self.normalize(self.data.velocities))
        color_palette = sns.color_palette('Paired', self.result.labels_.max() + 1 )
        cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in self.result.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, self.result.probabilities_)]
        plt.scatter(*projection.T, s=20, c=cluster_member_colors , linewidth=0, alpha=0.25)
    plt.show()
        
    def plot_cluster_distribution(self, colormap='Dark2'):
        v_count = dict()
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i) for i in np.linspace(0, 1, self.result.labels_.max() + 1)]
        for label in self.result.labels_:
            if label != -1:
                if label in v_count.keys():
                    v_count[label] += 1
                else:
                    v_count[label] = 1
        pd.Series({k: v for k, v in sorted(v_count.items(), key=lambda item: item[0])}).plot(kind='bar', color=colors)
        return v_count
    
    def get_data_from_class(self, num_label):
        data = []
        for i in range(len(self.result.labels_)):
            if self.result.labels_[i] == num_label:
                data.append(self.data[i])
        return data
    
    def plot_cluster_result(self, n_cols=3):
        labels = self.result.labels_
        n_clusters = labels.max() + 1
        n_rows = int(n_clusters / n_cols) if n_clusters % n_cols == 0 else int(math.ceil(n_clusters / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(25,15))
        for num_cluster in range(n_clusters):
            for serie_index in range(len(self.data.magnitudes)):
                if labels[serie_index] == num_cluster:
                    axs[int(num_cluster / n_cols), num_cluster % n_cols].plot(self.data.magnitudes[serie_index], c='blue', alpha=0.2)
            axs[int(num_cluster / n_cols), num_cluster % n_cols].set_title('Cluster %d'%(num_cluster + 1))
        fig.tight_layout()
        
    # créer un fichier csv contenant les champs : id, lat, lon, numero de la classe
    def save_result(self):

        fieldnames = ['id', 'Lat', 'Lon', 'cluster']
        rows = []
        
        for n in range(len(self.data.ns)):
            rows.append({'id': int(self.data.ns[n].id), 'Lat': self.data.ns[n].lat, 'Lon': self.data.ns[n].lon, 'cluster': self.result.labels_[n]})

        with open('clustering_result.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
    def generate_kml_file(self, scale=2):
        labels = self.result.labels_
        if labels.max() > -1:
            icon_links, scales = self.generate_icones(labels)
            kml=simplekml.Kml()
            fol = kml.newfolder(name="HDBSCAN Clustering")
            labels = self.result.labels_
            for i, n in enumerate(self.data.non_filtered_ids, 0):
                if labels[i] > - 1:
                    pnt = fol.newpoint(coords=[(self.data.pixels[n].lat, self.data.pixels[n].lon)])
                    pnt.iconstyle.icon.href =icon_links[labels[i]]
                    pnt.style.labelstyle.scale = scales[labels[i]]
            kml.save('clustering_result.kml')
        else:
            print('Hdbscan only found outliers. kml file cannot be generated !')
            
    def get_pixel_icon(self, m):
        labels = self.result.labels_
        icon_links, _ = self.generate_icones(labels)
        for i, n in enumerate(self.data.non_filtered_ids, 0):
            if n == m:
                return icon_links[labels[i]]
    
    #http://tancro.e-central.tv/grandmaster/markers/google-icons/mapfiles-ms-micons.html
    def generate_icones(self, labels, scale=2):
        BASE = 'http://maps.google.com/mapfiles/ms/micons/'
        scales = None
        colors = ['blue', 'red', 'yellow', 'green', 'orange', 'purple', 'pink']
        icon_links = [BASE + color + '-dot.png' for color in (colors + ['Itblue'])[:]] + [BASE + color + '.png' for color in (colors + ['lightblue'])[:]]
        n_clusters = labels.max() + 1
        if n_clusters > len(icon_links):
            q = int(n_clusters / len(icon_links))
            r = n_clusters - q*len(icon_links)
            icon_links = (icon_links*q)[:] + icon_links[:r]
            scales = [scale*t for t in range(len(icon_links))]
        else:
            scales = [scale]*n_clusters
        return icon_links, scales
    
    # coefficient de silhouette ?
    def validate(self):
        return DBCV(self.data.magnitudes, self.result.labels_, dist_function=euclidean)