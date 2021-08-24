from TimeSerie import TimeSerie
from Pixel import Pixel
from Data import Data
from Clustering import Clustering
from DataFromImageCorrelation import DataFromImageCorrelation
from Ica import Ica
import os
from osgeo import gdal

if __name__ == '__main__':

    ew_csv_path = 'donnees/Lavalette/MM_TIO_EW_31TGK_20151227_to_20200906.csv'
    ns_csv_path = 'donnees/Lavalette/MM_TIO_NS_31TGK_20151227_to_20200906.csv'
    slope_map_path = 'donnees/Lavalette/slope_lavalette.tif'
    azimuth_map_path = 'donnees/Lavalette/azimuth_lavalette.tif'
    ica_path = 'donnees/ica/'

    # data loading
    #print('loading data ...')
    #data = Data(ew_csv_path,  ns_csv_path, slope_map_path, azimuth_map_path)
    #image_correlation =  DataFromImageCorrelation(data.pixels[:100])

    # filtering
    #print('filtering data ...')
    # .filter_by(factor=2, azimuth_tol=100, azimuth_pr=0.2, min_slope=15, alpha=0.05)
    #print('computing mean velocity profiles ...')
    #image_correlation.compute_velocities()

    # clustering
    # print('clustering ...')
    # clustering_results = Clustering(image_correlation, option=0)
    # clustering_results.cluster(min_cluster_size=50, min_samples=10, cluster_selection_epsilon=None, precomputed=False)
    # print('generating kml file ...')
    #clustering_results.generate_kml_file_all()

    UU = []
    VV = []
    F2 = os.listdir(ica_path)
    for f in F2:
        if f.startswith('MM_EW') and f.endswith('.tif'):
            ds = gdal.Open(ica_path + f)
            img = ds.ReadAsArray()
            UU.append(img)
        if f.startswith('MM_NS') and f.endswith('.tif'):
            ds = gdal.Open(ica_path + f)
            img = ds.ReadAsArray()
            VV.append(img)

    print('complete UU and VV ')
    A = []
    n_filenames = UU.shape[0] * 2
    for i in range(0, n_filenames - 1):
        for j in range(i + 1, n_filenames):
            b = np.zeros(n_filenames - 1)
            b[i:j] = 1
            A.append(b)
    AA = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A))