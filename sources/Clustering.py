import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_regression
from statsmodels.tsa.stattools import adfuller
from sklearn.manifold import TSNE
import csv
import matplotlib.pyplot as plt
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
import hdbscan
from dtaidistance import dtw


class Clustering():
    """
    Cette classe a pour objectif de clusteriser les profils de vitesse moyenne en utilisant l'algorithme HDBSCAN
    Attributes :
        data : les profils de vitesse moyenne.
    """
    base = 'http://maps.google.com/mapfiles/ms/micons/'

    def __init__(self, data, option=0):
        self.data = data
        self.result = None
        self.n_clusters = None
        self.option = option

    def get_scaler(self):
        if self.option == 0:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        return scaler

    def normalize(self, series):
        """
        fonction qui normalise les données en utilisant la formule x = (x - x.mean()) / x.std()
        Args:
            series: liste de dataframes
                    profils de vitesse moyenne (issus de l'étape de filtrage)
        Returns:
            numpy array
                        les profils de vitesse normalisés
        """
        scaler = self.get_scaler()
        return np.array([scaler.fit_transform(serie).reshape(len(serie)) for serie in series])

    def compute_similarity_matrix(self, series):
        """
        fonction qui calcule une matrice de similarité en utilisant le distance élastique (DTW)
        Args:
            series:
        Returns:
        """
        velocities = self.data.velocities
        size = len(series)
        distances_matrix = np.zeros(shape=(size, size))
        scaler = StandardScaler()
        for n in range(size):
            for m in range(n, size):
                s1 = scaler.fit_transform(velocities[n])
                s2 = scaler.fit_transform(velocities[m])
                dist = dtw.distance(s1, s2)
                distances_matrix[n, m] = dist
                distances_matrix[m, m] = dist
        return distances_matrix

    def cluster(self, min_cluster_size, min_samples=None, cluster_selection_epsilon=None, precomputed=False):
        """
        fonction qui clusterise les données
        Args:
            min_cluster_size: taille minimum d'un cluster
            min_samples:  nombre d'échantillons mimimum (permet de contrôler la densité des clusters)
            cluster_selection_epsilon: permet de fusionner les clusters dont la distance est inférieure à un certain seuil
            precomputed: si ce paramètre vaut False alors on utilise directement les données pour le clustering, sinon on utilise une matrice de distance (exemple une matrice de similarité)
        Returns:
                None
        """
        data = self.data.velocities
        if not precomputed:
            self.result = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                          gen_min_span_tree=True).fit(self.normalize(data))
        else:
            self.result = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                          gen_min_span_tree=True, metric='precomputed').fit(
                self.compute_similarity_matrix(data))

    def get_n_clusters(self):
        return 0 if self.result.labels_.max() < 0 else self.result.labels_.max() + 1

    def visualize(self):
        projection = TSNE().fit_transform(self.normalize(self.data.velocities))
        color_palette = sns.color_palette('Paired', self.result.labels_.max() + 1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in self.result.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, self.result.probabilities_)]
        plt.scatter(*projection.T, s=20, c=cluster_member_colors, linewidth=0, alpha=0.25)
        plt.savefig('tsne.png')
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
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 15))
        for num_cluster in range(n_clusters):
            for serie_index in range(len(self.data.magnitudes)):
                if labels[serie_index] == num_cluster:
                    axs[int(num_cluster / n_cols), num_cluster % n_cols].plot(self.data.magnitudes[serie_index],
                                                                              c='blue', alpha=0.2)
            axs[int(num_cluster / n_cols), num_cluster % n_cols].set_title('Cluster %d' % (num_cluster + 1))
        fig.tight_layout()
        plt.show()

    # créer un fichier csv contenant les champs : id, lat, lon, numero de la classe
    def generate_csv_file(self):
        """
        fonction qui stocke les résultats du clustering dans un fichier csv
        Returns:
            fichier csv
                les résultats du clustering
        """
        rows = []
        fieldnames = ['id', 'Lat', 'Lon', 'cluster']
        data = self.data.pixels
        non_filtered_ids = self.data.non_filtered_ids
        for n in range(len(non_filtered_ids)):
            rows.append(
                {'id': int(data[n].id), 'Lat': data[n].lat, 'Lon': data[n].lon, 'cluster': self.result.labels_[n]})
        with open('clustering_result.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_important_labels(self, labels, n_classes=15):
        """
        fonction qui extrait les labels des clusters les plus importants. Le nombre de markers disponible pour afficher les données sur google earth est limité. J'ai donc décidé d'attribuer des markers
        aux clusters les plus volumineux. Les autres clusters seront représentés par le même marker.
        Args:
            labels:  les labels du clustering
            n_classes:  nombre de classes à extraire
        Returns:
                dict
        """
        classes = []
        unique, counts = np.unique(labels, return_counts=True)
        sort_labels = sorted(dict(zip(unique, counts)).items(), key=lambda x: x[1], reverse=True)
        for label, count in sort_labels:
            if label > -1 and len(classes) < n_classes:
                classes.append(label)
        return classes

    def generate_kml_file_all(self):
        labels = self.result.labels_
        latitudes = [self.data.pixels[n].lat for n in self.data.non_filtered_ids]
        longitudes = [self.data.pixels[n].lon for n in self.data.non_filtered_ids]
        if labels.max() > -1:
            icones = self.generate_icones2()
            dense_cluster_labels = self.get_important_labels(labels)
            kml = simplekml.Kml()
            fol = kml.newfolder(name="HDBSCAN Clustering")
            for n, v in enumerate(self.data.non_filtered_ids):
                pnt = None
                if labels[n] > -1:
                    pnt = fol.newpoint(description=str(v), coords=[(latitudes[n], longitudes[n])])
                    if labels[n] in dense_cluster_labels:
                        ind = dense_cluster_labels.index(labels[n])
                        pnt.iconstyle.icon.href = icones[ind]
                    else:
                        pnt.iconstyle.icon.href = icones[-1]
            kml.save('Resultats/clustering_results.kml')
            print('kml file created !')
        else:
            print('Hdbscan only found outliers. kml file cannot be generated !')

    def generate_kml_file2(self):
        if self.result.labels_.max() > -1:
            icones = self.generate_icones2()
            labels = self.get_important_labels(self.result.labels_)
            kml = simplekml.Kml()
            fol = kml.newfolder(name="HDBSCAN Clustering")
            for i, n in enumerate(self.data.non_filtered_ids, 0):
                if self.result.labels_[i] in labels:
                    ind = labels.index(self.result.labels_[i])
                    pnt = fol.newpoint(description=str(n), coords=[(self.data.pixels[n].lat, self.data.pixels[n].lon)])
                    pnt.iconstyle.icon.href = icones[ind]
            kml.save('clustering_result0.kml')
        else:
            print('Hdbscan only found outliers. kml file cannot be generated !')

    def generate_icones2(self):
        base = Clustering.base
        colors = ['blue', 'red', 'yellow', 'green', 'orange', 'purple', 'pink']
        return [base + color + '-dot.png' for color in (colors + ['ltblue'])[:]] + [base + color + '.png' for color in
                                                                                    (colors + ['lightblue'])[:]]
    def generate_kml_file(self, scale=2):
        """
        fonction qui stocke les résultats du clustering dans un fichier kml (qui sera ensuite visualisée à l'aide de Google Earth)
        Args:
            scale:
        Returns:
        """
        labels = self.result.labels_
        if labels.max() > -1:
            icon_links, scales = self.generate_icones(labels)
            kml = simplekml.Kml()
            fol = kml.newfolder(name="HDBSCAN Clustering")
            labels = self.result.labels_
            for i, n in enumerate(self.data.non_filtered_ids, 0):
                if labels[i] > - 1:
                    pnt = fol.newpoint(description=str(n), coords=[(self.data.pixels[n].lat, self.data.pixels[n].lon)])
                    pnt.iconstyle.icon.href = icon_links[labels[i]]
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

    def generate_icones2(self):
        base = Clustering.base
        colors = ['blue', 'red', 'yellow', 'green', 'orange', 'purple', 'pink']
        return [base + color + '-dot.png' for color in (colors + ['ltblue'])[:]] + [base + color + '.png' for color in
                                                                                    (colors + ['lightblue'])[:]]