from TimeSerie import TimeSerie
from Pixel import Pixel
from Data import Data
from Clustering import Clustering
from DataFromImageCorrelation import DataFromImageCorrelation


if __name__ == '__main__':

    ew_csv_path = 'rasters/Lavalette/MM_TIO_EW_31TGK_20151227_to_20200906.csv'
    ns_csv_path = 'rasters/Lavalette/MM_TIO_NS_31TGK_20151227_to_20200906.csv'
    slope_map_path = 'rasters/Lavalette/slope_lavalette.tif'
    azimuth_map_path = 'rasters/Lavalette/azimuth_lavalette.tif'

    # data loading
    data = Data(ew_csv_path,  ns_csv_path, slope_map_path, azimuth_map_path)

    # filtering
    image_correlation =  DataFromImageCorrelation(data.pixels)
    image_correlation .filter_by(factor=2, azimuth_tol=100, azimuth_pr=0.2, min_slope=15, alpha=0.05)
    image_correlation.compute_velocities()

    # clustering
    clustering_results = Clustering(image_correlation, option=0)
    clustering_results.cluster(min_cluster_size=50, min_samples=10, cluster_selection_epsilon=None, precomputed=False)
    clustering_results.generate_kml_file_all()

