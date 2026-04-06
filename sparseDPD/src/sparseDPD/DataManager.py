# Create a Dataset object for loading and processing data
import numpy as np
import scipy
from .Dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt

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
            self.training_dataset = Dataset(train_input, train_output)
            val_input =     self._iq_to_complex(val_input_df)
            val_output =    self._iq_to_complex(val_output_df)
            self.validation_dataset = Dataset(val_input, val_output)
            test_input =    self._iq_to_complex(test_input_df)
            test_output =   self._iq_to_complex(test_output_df)
            self.test_dataset = Dataset(test_input, test_output)
            self.input_data = np.concatenate([train_input, val_input, test_input]) 
            self.output_data = np.concatenate([train_output, val_output, test_output])  

        elif filepath:
            self.filepath = filepath
            self.input_data, self.output_data = self.read_file()
        else:
            raise ValueError("Either filepath or all OpenDPD file paths must be provided.")

        if not openDPD_test_input_file or not openDPD_test_output_file or not openDPD_training_input_file or not openDPD_training_output_file or not openDPD_validation_input_file or not openDPD_validation_output_file:
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
            # Check for 'x' and 'y' columns first, fallback to 'x1' and 'u1'
            if 'x' in data and 'y' in data:
                self.input_data = data['x'].squeeze()
                self.output_data = data['y'].squeeze()
            elif 'x1' in data and 'u' in data:
                self.input_data = data['x1'].squeeze()
                self.output_data = data['u'].squeeze()
            else:
                raise KeyError(f"Could not find expected data columns in .mat file. "
                             f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}")
        return self.input_data, self.output_data
    
    def plot_pa_characteristics(self, num_points=5000, figsize=(14, 6)):
        """Plot AM/AM and AM/PM characteristics of the PA.
        
        Shows the nonlinear behavior of the power amplifier by plotting:
        - AM/AM: Output amplitude vs Input amplitude (gain compression)
        - AM/PM: Output phase shift vs Input amplitude (phase distortion)
        
        Parameters:
        -----------
        num_points : int, optional
            Number of data points to plot (for performance). Default is 5000.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (14, 6).
        """

        input_signal = self.input_data
        output_signal = self.output_data
        # Downsample if dataset is large
        if len(self.input_data) > num_points:
            indices = np.linspace(0, len(self.input_data) - 1, num_points, dtype=int)
            input_signal = self.input_data[indices]
            output_signal = self.output_data[indices]
        else:
            input_signal = self.input_data
            output_signal = self.output_data
        
        # Calculate amplitudes
        input_amplitude = np.abs(input_signal)
        output_amplitude = np.abs(output_signal)
        
        # Calculate phase difference (AM/PM characteristic)
        input_phase = np.angle(input_signal)
        output_phase = np.angle(output_signal)
        phase_diff = np.unwrap(output_phase - input_phase)  # Unwrap to avoid discontinuities
        phase_diff_deg = np.degrees(phase_diff)  # Convert to degrees
        phase_diff_deg = phase_diff_deg - np.mean(phase_diff_deg)  # Center around 0
        
        # Sort by input amplitude for cleaner visualization
        sort_idx = np.argsort(input_amplitude)
        input_amp_sorted = input_amplitude[sort_idx]
        output_amp_sorted = output_amplitude[sort_idx]
        phase_diff_sorted = phase_diff_deg[sort_idx]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # AM/AM Plot
        ax1.scatter(input_amp_sorted, output_amp_sorted, alpha=0.3, s=1, c='blue', label='PA Behavior')
        
        # Ideal AM/AM (linear, unity gain)
        amp_range = np.array([0, np.max(input_amp_sorted)])
        # Estimate approximate gain from data for ideal line
        estimated_gain = np.median(output_amp_sorted[input_amp_sorted > 0] / input_amp_sorted[input_amp_sorted > 0])
        ideal_output = amp_range * estimated_gain
        ax1.plot(amp_range, ideal_output, 'k--', linewidth=2, alpha=0.7, label='Ideal (Linear)')
        
        ax1.set_xlabel('Input Amplitude', fontsize=20)
        ax1.set_ylabel('Output Amplitude', fontsize=20)
        ax1.set_title('AM/AM Characteristic\n(Gain Compression)', fontsize=22, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=18)
        
        # AM/PM Plot
        ax2.scatter(input_amp_sorted, phase_diff_sorted, alpha=0.3, s=1, c='red', label='PA Behavior')
        
        # Ideal AM/PM (no phase distortion)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (No Phase Shift)')
        
        ax2.set_xlabel('Input Amplitude', fontsize=20)
        ax2.set_ylabel('Phase Shift (degrees)', fontsize=20)
        ax2.set_title('AM/PM Characteristic\n(Phase Distortion)', fontsize=22, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=18)

        # Limit the AM/PM plot to a reasonable range for better visualization
        ax2.set_ylim(-180, 180)

        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== PA Characteristic Summary ===")
        print(f"Input amplitude range: {np.min(input_amplitude):.4f} to {np.max(input_amplitude):.4f}")
        print(f"Output amplitude range: {np.min(output_amplitude):.4f} to {np.max(output_amplitude):.4f}")
        print(f"Estimated small-signal gain: {estimated_gain:.4f}")
        print(f"Phase distortion range: {np.min(phase_diff_deg):.2f}° to {np.max(phase_diff_deg):.2f}°")


    def save_to_mat_file(self, output_data, filename):
        """Save input and output data to a .mat file with the same structure as the original."""
        scipy.io.savemat(filename, {'x': self.input_data, 'y': output_data})