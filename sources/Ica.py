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
from sklearn.cluster import DBSCAN
import numpy.ma as ma
from numpy import linalg as LA
import random
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats


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

    def generate_components(self, n_components_min=2, n_components_max=30, step=1, max_iter=100, ica_max_iter=300,
                            seed=1000, max_value=1000):
        random.seed(seed)
        random_states = random.sample(range(max_value), max_iter)
        uu_components, vv_components = [], []

        for n_components in range(n_components_min, n_components_max, step):
            for n_iter in range(max_iter):
                uu_ica = FastICA(n_components=n_components, random_state=random_states[n_iter], max_iter=ica_max_iter)
                vv_ica = FastICA(n_components=n_components, random_state=random_states[n_iter], max_iter=ica_max_iter)
                uu_components_iter = np.transpose(uu_ica.fit_transform(self.uu_4ica))
                vv_components_iter = np.transpose(vv_ica.fit_transform(self.vv_4ica))
                for rw in np.transpose(uu_ica.mixing_):
                    uu_components.append(list(rw))
                for rw in np.transpose(vv_ica.mixing_):
                    vv_components.append(list(rw))

        return np.array(uu_components), np.array(vv_components)

    def compute_similarity_matrix(data, metric='pearson'):
        size = len(data)
        distances_matrix = np.zeros(shape=(size, size))

        for n in range(size):
            for m in range(n + 1, size):
                r = 1 - abs(pearsonr(data[n], data[m])[0])  # Est-ce vraiment pertient de considérer la vakeur absolue ?
                distances_matrix[m, n] = r
                distances_matrix[n, m] = r

    def average_intra_cluster_similarity(self, data, labels, num_cluster):
        """
        fonction qui calcule la similarité intra-cluster
        parameters
        ----------
        data : numpy ndarray
            les composantes indépendantes stockés ligne par ligne
        labels : numpy array
            les labels du clustering
        num_cluster : int
            le numéro du cluster

        Returns
        -------
            float
                la similarité intra-cluster du cluster de numéro num_cluster

        """
        r = 0.0
        object_indexes = np.where(labels == num_cluster)[0]
        card_cluster = object_indexes.shape[0]
        for i in range(card_cluster):
            for j in range(i, card_cluster):
                n, m = object_indexes[i], object_indexes[j]
                r += abs(pearsonr(data[n], data[m])[0])

        return r / (card_cluster * card_cluster)

    def average_inter_cluster_similarity(self,data, labels, num_cluster):
        """
        fonction qui calcule la similarité inter-cluster

        parameters
        ----------
        data : numpy ndarray
        labels : numpy array
            les labels du clustering
        num_cluster : int
            numéro du cluster

        Returns
        -------
            float
                similarité inter-cluster du cluster de numéro num_cluster
        """
        r = 0.0
        object_indexes = np.where(labels == num_cluster)[0]
        non_object_indexes = np.where(labels != num_cluster)[0]
        card_cluster = object_indexes.shape[0]

        for i in range(object_indexes.shape[0]):
            for j in range(non_object_indexes.shape[0]):
                n, m = object_indexes[i], non_object_indexes[j]
                r += abs(pearsonr(data[n], data[m])[0])

        return r / (card_cluster * (labels.shape[0] - card_cluster))

    def cluster_quality(self, data, labels, num_cluster):
        """
        fonction qui calcule la stabilité d'un cluster : différence entre la similarité intra-cluster et la similarité inter-cluster

        parameters
        ----------
        data : numpy ndarray
            composantes indépendantes stockées ligne par ligne
        labels : numpy array
            labels du clustering
        num_cluster : int
            numéro du cluster

        Returns
        -------
            float
                stabilité d'un cluster
        """

        avg_inter_sm = self.average_inter_cluster_similarity(data, labels, num_cluster)
        avg_intra_sm = self.average_intra_cluster_similarity(data, labels, num_cluster)
        return avg_intra_sm - avg_inter_sm

    def average_clustering_quality(self, data, labels, n_clusters):
        """
        parameters
        ----------
        data : numpy ndarray
            les composantes indépendantes stockées ligne par lignes
        labels : numpy array
            labels du clustering
        n_clusters : int
            nombre de clusters

        Returns
        -------
        """
        avg = 0.0
        for num_cluster in range(n_clusters):
            avg += self.cluster_quality(data, labels, num_cluster)
        return avg / n_clusters

    # do not use that fonction
    def validate(self, data, n_components_min, n_components_max, n_iterations):
        start = 0
        pearson, silhouette = [], []

        for n_components in range(n_components_min, n_components_max, 1):
            end = start + n_components * n_iterations
            X = data[start:end, :]
            print(X.shape)
            distances = self.compute_similarity_matrix(X)
            results = AgglomerativeClustering(affinity='precomputed', n_clusters=n_components, linkage='average').fit(
                distances)
            pearson.append(self.average_clustering_quality(X, results.labels_, n_components))
            silhouette.append(silhouette_score(distances, results.labels_, metric='precomputed'))
            print('%d finished' % n_components)
            start = end + 1

        return pearson, silhouette

    def validate(self, data, n_components_min, n_components_max, n_iterations, ward=True):
        """
        fonction qui applique des indices de validation externe aux jeu de données composé  de composantes indépendantes

        parameters
        ----------
        data : numpy ndarray
            composantes indépendantes stcokées ligne par ligne

        n_components_min : int
            nombre de composantes indépendantes minimum

        n_components_max : int
            nombre de composantes indépendantes maximum

        n_iterations : int
            nombre d'iterations

        ward : bool
            si True alors on utilise la distance euclidenne pour comparer les sources sinon on utilise on utillise l'indice de validation
            définit par les auteurs de ICASSO

        Returns
        -------
            list, list
            valeurs du coefficient de l'indice ICASSO, valeurs du coefficient de silhouette
        """
        start = 0
        pearson, silhouette = [], []

        for n_components in range(n_components_min, n_components_max, 1):
            end = start + n_components * n_iterations
            X = data[start:end, :]
            distances = self.compute_similarity_matrix(X)
            results = None
            if not ward:
                results = AgglomerativeClustering(affinity='precomputed', n_clusters=n_components,
                                                  linkage='average').fit(distances)
            else:
                results = AgglomerativeClustering(n_clusters=n_components).fit(X)
                print('ward')
            pearson.append(self.average_clustering_quality(X, results.labels_, n_components))
            silhouette.append(silhouette_score(distances, results.labels_, metric='precomputed'))
            print('%d finished' % n_components)
            start = end + 1

        return pearson, silhouette

if __name__ == "__main__":

    # X1 = Ica.generate_components(uu_4ica, n_components_min=2, n_components_max=30, step=1, max_iter=60)
    # X2 = Ica.generate_components(vv_4ica, n_components_min=2, n_components_max=30, step=1, max_iter=60)


