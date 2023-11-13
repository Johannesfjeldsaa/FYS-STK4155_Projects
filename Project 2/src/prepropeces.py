import os
#import xarray as xr
import matplotlib.pyplot as plt

class Handle_Files:

    def __init__(self):
        self.working_dir = os.getcwd()

        # Where to save the figures and data files
        self.project_results_dir = self.working_dir+" Results"
        self.results_figure_dir = self.working_dir+" Results/FigureFiles"
        self.data_dir = self.working_dir+" DataFiles/"

        if not os.path.exists(self.project_results_dir):
            os.mkdir(self.project_results_dir)

        if not os.path.exists(self.results_figure_dir):
            os.makedirs(self.results_figure_dir)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def image_path(self, fig_id):
        return os.path.join(self.results_figure_dir, fig_id)

    def data_path(self, dat_id):
        return os.path.join(self.data_dir, dat_id)

    def save_fig(self, fig_id):
        plt.savefig(self.image_path(fig_id) + ".png", format='png')


    def get_all_filenames_in_dir(self, directory, condition=None):
        """
        Get all the filenames in a directory that satisfy a condition.

        Parameters:
        - directory (str): Path to the directory where the files are located.
        - condition (function): A function that takes a filename as input and
        returns a boolean value indicating whether the filename satisfies the
        condition.

        Returns:
        - list: A list of filenames that satisfy the condition.
        """
        filenames = os.listdir(directory)
        if condition is None:
            return filenames
        else:
            return [filename for filename in filenames if condition(filename)]


    def get_all_netcdf_files_in_dir(self, directory):
        """
        Get all the NetCDF files in a directory.

        Parameters:
        - directory (str): Path to the directory where the files are located.

        Returns:
        - list: A list of filenames that satisfy the condition.
        """
        return self.get_all_filenames_in_dir(directory, condition=lambda filename: filename.endswith(".nc"))
