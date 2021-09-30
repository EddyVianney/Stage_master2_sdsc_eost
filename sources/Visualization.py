import matplotlib.pyplot as plt
import simplekml
import numpy as np

class Visualization():

    def __init__(self):
        pass

    def plot_disp_ns_ew(self, pixel, title):

        fig, ax = plt.subplots(3, 1, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.2)

        ax[0].plot(pixel.ns, color='blue', marker='o', linewidth=2)
        ax[0].set_title('déplacements cumulés nord-sud', fontsize=18)
        # ax[0].set_xlabel('dates d\'acquisitions')
        ax[0].set_ylabel('déplacement (m)', fontsize=13)

        ax[1].plot(pixel.ew, color='orange', marker='o', linewidth=2)
        ax[1].set_title('déplacements cumulés est-ouest', fontsize=18)
        # ax[1].set_xlabel('dates d\'acquisitions')
        ax[1].set_ylabel('déplacement (m)', fontsize=13)

        ax[2].plot(pixel.azimuths, color='green', marker='o', linewidth=2)
        ax[2].set_title('variations de l\' azimuth', fontsize=18)
        ax[2].set_xlabel('dates d\'acquisition', fontsize=13)
        ax[2].set_ylabel('azimuth (°)', fontsize=13)

        plt.savefig(title + '.png')
        fig.tight_layout()
        plt.show()

    def plot_disp_vel(self, ns_ts, ew_ts, vels):

        fig, ax = plt.subplots(3, 1, figsize=(15, 10))

        ax[0].plot(ns_ts, color='blue', label='displacement (m)', marker='o', linewidth=2)
        ax[0].set_title('NS cumulative displacement')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('displacement')
        ax[0].legend()

        ax[1].plot(ew_ts, color='orange', label='displacement (m)', marker='o', linewidth=2)
        ax[1].set_title('EW cumulative displacement')
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('displacement')
        ax[1].legend()

        ax[2].plot(vels, color='green', label='vxelocity (m/day)', marker='o', linewidth=2)
        ax[2].set_title('Velocity magnitude')
        ax[2].set_xlabel('time')
        ax[2].set_ylabel('velocity')
        ax[2].legend()

        fig.tight_layout()
        plt.show()

    def plot_series(series, num_rows=48677, num_cols=5, colormap='tab20'):

        plot_kwds = {'alpha': 0.25, 's': 10, 'linewidths': 0}
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 25))
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i) for i in np.linspace(0, 1, num_rows * num_cols)]

        for num_row in range(num_rows):
            for num_col in range(num_cols):
                if num_row * num_cols + num_col < len(series):
                    axs[num_row, num_col].plot(series[num_row * num_cols + num_col],
                                               color=colors[num_row * num_cols + num_col], marker='o',
                                               markerfacecolor='white')
                    # axs[num_row, num_col].set_title('serie: %s'%(self.names[num_row*num_cols + num_col].split('.')[0]))
        plt.show()

    def generate_km_file_testing(self, pixels, listes, colors):
        url = 'http://maps.google.com/mapfiles/ms/micons/'
        kml = simplekml.Kml()
        for m in range(len(listes)):
            for n in listes[m]:
                lat, lon = pixels[n].lat, pixels[n].lon
                pnt = kml.newpoint(description=str(n), coords=[(lat, lon)])
                pnt.iconstyle.icon.href = url + colors[m] + '-dot.png'
        kml.save('testing_points' + '.kml')

    def generate_kml_file(self, mask, latitudes, longitudes, color, filename='visualization'):

        url = 'http://maps.google.com/mapfiles/ms/micons/'
        kml = simplekml.Kml()
        for n, coords in enumerate(zip(latitudes, longitudes)):
            if not mask[n]:
                pnt = kml.newpoint(description=str(n), coords=[(coords[0], coords[1])])
                pnt.iconstyle.icon.href = url + color + '-dot.png'
        kml.save(filename + '.kml')

    def generate_kml_file3(self, mask, latitudes, longitudes, color, filename='visualization'):

        url = 'http://maps.google.com/mapfiles/ms/micons/'
        kml = simplekml.Kml()
        for n in mask:
            latitude = latitudes[n]
            longitude = longitudes[n]
            pnt = kml.newpoint(description=str(n), coords=[(latitude, longitude)])
            pnt.iconstyle.icon.href = url + color + '-dot.png'
        kml.save(filename + '.kml')