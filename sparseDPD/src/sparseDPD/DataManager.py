# Create a Dataset object for loading and processing data
import numpy as np
import scipy
from .Dataset import Dataset
import pandas as pd

class DataManager:
    def __init__(self, num_training_points, num_validaiton_points, num_test_points, filepath=None,
                 openDPD_test_input_file=None, openDPD_test_output_file=None, openDPD_training_input_file=None, openDPD_training_output_file=None, openDPD_validation_input_file=None, openDPD_validation_output_file=None):
        """Breaks full data set into section"""

        if openDPD_test_input_file and openDPD_test_output_file and openDPD_training_input_file and openDPD_training_output_file and openDPD_validation_input_file and openDPD_validation_output_file:
            # Create the input and output vectors, the files are all csv files and contain to columns I and Q
            train_input_df = pd.read_csv(openDPD_training_input_file)
            train_output_df = pd.read_csv(openDPD_training_output_file)
            val_input_df = pd.read_csv(openDPD_validation_input_file)
            val_output_df = pd.read_csv(openDPD_validation_output_file)
            test_input_df = pd.read_csv(openDPD_test_input_file)
            test_output_df = pd.read_csv(openDPD_test_output_file)

            # Create complex arrays for all datasets
            train_input =   self._iq_to_complex(train_input_df)
            train_output =  self._iq_to_complex(train_output_df)
            val_input =     self._iq_to_complex(val_input_df)
            val_output =    self._iq_to_complex(val_output_df)
            test_input =    self._iq_to_complex(test_input_df)
            test_output =   self._iq_to_complex(test_output_df)
            self.input_data = np.concatenate([train_input, val_input, test_input]) 
            self.output_data = np.concatenate([train_output, val_output, test_output])  
        elif filepath:
            self.filepath = filepath
            self.input_data, self.output_data = self.read_file()
        else:
            raise ValueError("Either filepath or all OpenDPD file paths must be provided.")


        self.num_training_points = num_training_points
        self.num_validaiton_points = num_validaiton_points
        self.num_test_points = num_test_points

        self.valid_index = None
        self.test_index = None

        self.training_dataset = self.get_training_data()
        self.validation_dataset = self.get_validation_data()
        self.test_dataset = self.get_test_data()

    def _iq_to_complex(self, df):
        """Convert DataFrame with I and Q columns to complex numpy array"""
        return (df['I'].values + 1j * df['Q'].values).astype(np.complex128)

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
    
