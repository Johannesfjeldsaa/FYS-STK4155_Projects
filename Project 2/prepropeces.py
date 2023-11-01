import os
import xarray as xr
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

    def read_netcdf_to_xr(self, directory, file_name):
        """
        Read a NetCDF file and return its content as an xarray dataset.

        Parameters:
        - file_path (str): Path to the NetCDF file to be read.

        Returns:
        - xr.Dataset: An xarray dataset containing the data from the NetCDF file.
        """
        try:
            file_path = os.path.join(directory, file_name)
            dataset = xr.open_dataset(file_path)
            return dataset
        except Exception as e:
            print(f"Error reading the NetCDF file: {e}")
            return None

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

class Preprocess_Data:

    def __init__(self):
        self.file_handler = Handle_Files()

    def preprocess_data(self, directory, file_name):

        pass
        """
        Preprocess the data in a NetCDF file.

        Parameters:
        - file_path (str): Path to the NetCDF file to be read.

        Returns:
        - xr.Dataset: An xarray dataset containing the preprocessed data.
        """
        dataset = self.file_handler.read_netcdf_to_xr(directory, file_name)
        if dataset is None:
            return None

        dataset = self.remove_outliers(dataset)
        dataset = self.remove_missing_values(dataset)
        dataset = self.remove_redundant_variables(dataset)
        dataset = self.remove_redundant_dimensions(dataset)
        dataset = self.remove_redundant_attributes(dataset)
        dataset = self.rename_dimensions(dataset)
        dataset = self.rename_variables(dataset)
        dataset = self.rename_attributes(dataset)
        dataset = self.convert_units(dataset)

        return dataset

    def remove_outliers(self, dataset):
        """
        Remove outliers from the dataset.

        Parameters:
        - dataset (xr.Dataset): An xarray dataset.

        Returns:
        - xr.Dataset: An xarray dataset with the outliers removed.
        """
        pass