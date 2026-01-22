# Create a Dataset object for loading and processing data
import numpy as np
import scipy
from Dataset import Dataset

class DataManager:
    def __init__(self, filepath, num_training_points, num_validaiton_points, num_test_points):
        """Breaks full data set into section"""

        self.filepath = filepath
        self.input_data, self.output_data = self.read_file()


        self.num_training_points = num_training_points
        self.num_validaiton_points = num_validaiton_points
        self.num_test_points = num_test_points

        self.valid_index = None
        self.test_index = None

        self.training_dataset = self.get_training_data()
        self.validation_dataset = self.get_validation_data()
        self.test_dataset = self.get_test_data()

    def get_training_data(self):
        self.valid_index = self.num_training_points
        return Dataset(self.input_data[:self.num_training_points], self.output_data[:self.num_training_points])
    
    def get_validation_data(self):
        self.test_index = self.valid_index + self.num_validaiton_points
        return Dataset(self.input_data[self.valid_index:self.test_index], self.output_data[self.valid_index:self.test_index])
    
    def get_test_data(self):
        return Dataset(self.input_data[self.test_index:], self.output_data[self.test_index:])
    
    def read_file(self):
        """Read input and output data from file"""
        if self.filepath.endswith(".mat"):
            data = scipy.io.loadmat(self.filepath)
            self.input_data = data['x'].squeeze()
            self.output_data = data['y'].squeeze()
        return self.input_data, self.output_data
    
