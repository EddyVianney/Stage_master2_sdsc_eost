from TimeSerie import TimeSerie
from Pixel import Pixel
from Data import Data
from Clustering import Clustering
from DataFromImageCorrelation import DataFromImageCorrelation
from Ica import Ica
import os
from osgeo import gdal

if __name__ == '__main__':

    # chemin du fichier csv contenant les déplacements Est-Ouest
    ew_csv_path = './donnees/Lavalette/MM_TIO_EW_31TGK_20151227_to_20200906.csv'
    # chemin du fichier csv contenant les déplacements Nord-Sud
    ns_csv_path = './donnees/Lavalette/MM_TIO_NS_31TGK_20151227_to_20200906.csv'
    # chemin du fichier raster pente (ce fichier a été généré manuellement en utilisant qgis)
    slope_map_path = './donnees/Lavalette/slope_lavalette.tif'
    # chemin du fichier raster azimut (ce fichier a été généré manuellement en utilisant qgis)
    azimuth_map_path = './donnees/Lavalette/azimuth_lavalette.tif'
    # chemin des données pour ICA
    ica_path = './donnees/ica/'

    # data loading
    print('loading data ...')
    data = Data(ew_csv_path,  ns_csv_path, slope_map_path, azimuth_map_path)
    image_correlation =  DataFromImageCorrelation(data.pixels)

    # filtering
    print('filtering data ...')
    image_correlation.filter_by(factor=None, azimuth_tol=None, azimuth_pr=None, min_slope=None, alpha=None, use_kurtosis=True)
    print('computing mean velocity profiles ...')
    image_correlation.compute_velocities()

    # clustering
    print('clustering ...')
    clustering_results = Clustering(image_correlation, option=0)
    clustering_results.cluster(min_cluster_size=60, min_samples=1, cluster_selection_epsilon=None, precomputed=False)
    print('generating kml file ...')
    clustering_results.generate_kml_file_all()