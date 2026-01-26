# Create a Dataset object for loading and processing data
import numpy as np
import scipy

class Dataset:
    def __init__(self, input_data, output_data):
        """Input and output data should be in complex format"""
        self.input_data = input_data
        self.output_data = output_data

    def input_phase(self):
        """Return the phase of a complex signal"""
        Ax = np.abs(self.input_data)
        return np.conj(self.input_data)/Ax

    def calculate_nmse(self):
        """Calculate NMSE between input and output signals"""
        power_input = np.mean(np.abs(self.input_data)**2)
        power_error = np.mean(np.abs(self.input_data - self.output_data)**2)
        nmse = 10 * np.log10(power_error / power_input)
        return nmse
    
    @staticmethod
    def conj_phase(signal):
        """Return the conj of phase of a complex signal"""
        A = np.abs(signal)
        return np.conj(signal)/A