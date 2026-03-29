# Full datapath with forward and inverse models (could be either NN or volterra)


from .Volterra import Volterra
from .Dataset import Dataset
from .NeuralNetwork import NeuralNetwork
from .PGJANET import PGJANET_NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np


class Datapath:
    def __init__(self, forward_model, inverse_model):
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def _get_model_trim_amount(self, model):
        """Get the number of samples trimmed by a model's output.
        PGJANET uses windowing and trims (seq_len - 1) samples.
        Other models trim num_memory_levels samples."""
        if isinstance(model, PGJANET_NeuralNetwork):
            return model.seq_len - 1
        elif isinstance(model, NeuralNetwork):
            return model.num_memory_levels
        elif isinstance(model, Volterra):
            return model.num_memory_levels
        else:
            return 0

    def process(self, input_signal):
        """Process the input signal through the inverse model followed by the forward model"""
        if type(self.inverse_model) == Volterra:
            pre_distorted_signal = self.inverse_model.build_y(input_signal)
            # Process through forward model
            if type(self.forward_model) == Volterra:
                output_signal = self.forward_model.build_y(pre_distorted_signal)
                total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
                input_signal = input_signal[total_trim:]  
            elif isinstance(self.forward_model, NeuralNetwork):
                output_signal = self.forward_model.generate_model_output(pre_distorted_signal)
                total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
                input_signal = input_signal[total_trim:]  # Align input signal
        elif isinstance(self.inverse_model, NeuralNetwork):
            pre_distorted_signal = self.inverse_model.generate_model_output(input_signal)
            if isinstance(self.forward_model, NeuralNetwork):
                output_signal = self.forward_model.generate_model_output(pre_distorted_signal)
                total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
                input_signal = input_signal[total_trim:]  # Align input signal
            elif type(self.forward_model) == Volterra:
                output_signal = self.forward_model.build_y(pre_distorted_signal)
                total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
                input_signal = input_signal[total_trim:]  # Align input signal
        else: 
            print(f"Your inverse model is type {type(self.inverse_model)} and your forward model is type {type(self.forward_model)}")
        # Trim input signal to line up with output signal and return both 
        dataset = Dataset(input_signal, output_signal)
        return dataset
    
    def train_using_ila(self, training_dataset, valid_dataset, iterations, retrain_epochs_per_iteration):
        # Generate initial outputs - no extra slicing needed, models handle trimming internally
        inverse_model_output = self.inverse_model.generate_model_output(training_dataset.input_data)
        forward_model_output = self.forward_model.generate_model_output(inverse_model_output)
        total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)

        for iteration in range(iterations):
            # Retrain inverse model on the error
            # Create dataset with forward model output vs original input (aligned)
            aligned_input = inverse_model_output[self._get_model_trim_amount(self.forward_model):]  # Align with forward model output
            new_dataset = Dataset(input_data=aligned_input, output_data=forward_model_output)
            train_losses_inv, valid_losses_inv, best_epoch_inv = self.inverse_model.get_best_model(
                num_epochs=retrain_epochs_per_iteration, 
                training_dataset=new_dataset,  # Use the new dataset, not original
                validation_dataset=valid_dataset
            )

            # Get new outputs after retraining
            inverse_model_output = self.inverse_model.generate_model_output(training_dataset.input_data)
            forward_model_output = self.forward_model.generate_model_output(inverse_model_output)

            # Print NMSE after this iteration
            dataset = Dataset(training_dataset.input_data[total_trim:], forward_model_output)  # Align input with output
            nmse = dataset.calculate_nmse()
            print(f"Iteration {iteration+1}/{iterations} - NMSE: {nmse:.4f} dB")

    def train(self, training_dataset, valid_dataset, epochs):
        
        train_losses_inv, valid_losses_inv, best_epoch_inv = self.inverse_model.get_best_model(
                num_epochs=epochs, 
                training_dataset=training_dataset,  # Use the new dataset, not original
                validation_dataset=valid_dataset
            )
        return train_losses_inv, valid_losses_inv, best_epoch_inv

               
    def calculate_nmse(self, input_signal):
        data = self.process(input_signal=input_signal)
        # Calculate NMSE
        return data.calculate_nmse()
    
    def plot_spectrum(self, input_data):
        """Should plot a comparison between the spectrum, and what it would be with no DPD"""
        output_data = self.process(input_data)
        output_data_no_DPD = self.forward_model.generate_model_output(input_data)
        plt.figure()
        
        plt.magnitude_spectrum(output_data.output_data, Fs=1, scale='dB', label='Output Signal')
        plt.magnitude_spectrum(output_data_no_DPD, Fs=1, scale='dB', label='Output without DPD')
        plt.title('Magnitude Spectrum of Input and Output Signals')
        plt.xlabel('Normalized Frequency (cycles/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid()
        plt.show()
    
    def calculate_aclr(self, input_data, channel_bw=0.1, channel_spacing=0.15, fs=1.0):
        # Get output with and without DPD
        output_data = self.process(input_data)
        output_data_no_DPD = self.forward_model.generate_model_output(input_data)
        
        # Calculate ACLR for both signals
        aclr_lower_dpd, aclr_upper_dpd = self._compute_aclr(
            output_data.output_data, channel_bw, channel_spacing, fs
        )
        aclr_lower_no_dpd, aclr_upper_no_dpd = self._compute_aclr(
            output_data_no_DPD, channel_bw, channel_spacing, fs
        )
        
        return {
            'aclr_lower_dpd': aclr_lower_dpd,
            'aclr_upper_dpd': aclr_upper_dpd,
            'aclr_lower_no_dpd': aclr_lower_no_dpd,
            'aclr_upper_no_dpd': aclr_upper_no_dpd
        }
    
    @staticmethod
    def _compute_aclr(signal, channel_bw, channel_spacing, fs):
        """
        Compute ACLR for a given signal.
        
        Returns:
        --------
        tuple : (lower_aclr, upper_aclr) in dBc
        """
        # Compute PSD using FFT
        N = len(signal)
        fft_signal = np.fft.fft(signal)
        psd = np.abs(fft_signal) ** 2 / N
        freq = np.fft.fftfreq(N, d=1/fs)
        
        # Shift to center DC at middle
        psd_shifted = np.fft.fftshift(psd)
        freq_shifted = np.fft.fftshift(freq)
        
        # Define channel boundaries
        main_lower = -channel_bw / 2
        main_upper = channel_bw / 2
        
        lower_adj_center = -channel_spacing
        lower_adj_lower = lower_adj_center - channel_bw / 2
        lower_adj_upper = lower_adj_center + channel_bw / 2
        
        upper_adj_center = channel_spacing
        upper_adj_lower = upper_adj_center - channel_bw / 2
        upper_adj_upper = upper_adj_center + channel_bw / 2
        
        # Integrate power in each channel
        main_mask = (freq_shifted >= main_lower) & (freq_shifted <= main_upper)
        lower_adj_mask = (freq_shifted >= lower_adj_lower) & (freq_shifted <= lower_adj_upper)
        upper_adj_mask = (freq_shifted >= upper_adj_lower) & (freq_shifted <= upper_adj_upper)
        
        main_power = np.sum(psd_shifted[main_mask])
        lower_adj_power = np.sum(psd_shifted[lower_adj_mask])
        upper_adj_power = np.sum(psd_shifted[upper_adj_mask])
        
        # Calculate ACLR in dBc (relative to carrier)
        # Add small epsilon to avoid log(0)
        eps = 1e-20
        aclr_lower = 10 * np.log10((main_power + eps) / (lower_adj_power + eps))
        aclr_upper = 10 * np.log10((main_power + eps) / (upper_adj_power + eps))
        
        return aclr_lower, aclr_upper
    
    def plot_spectrum_with_aclr(self, input_data, channel_bw=0.1, channel_spacing=0.15, fs=1.0):
        """
        Plot spectrum with ACLR channel boundaries marked.
        
        Parameters same as calculate_aclr()
        """
        # Get output with and without DPD
        output_data = self.process(input_data)
        output_data_no_DPD = self.forward_model.generate_model_output(input_data)
        
        # Calculate ACLR
        aclr_metrics = self.calculate_aclr(input_data, channel_bw, channel_spacing, fs)
        
        # Compute PSD for plotting - handle different signal lengths
        N_dpd = len(output_data.output_data)
        fft_dpd = np.fft.fft(output_data.output_data)
        psd_dpd = 10 * np.log10(np.abs(fft_dpd) ** 2 / N_dpd + 1e-20)
        freq_dpd = np.fft.fftfreq(N_dpd, d=1/fs)
        
        N_no_dpd = len(output_data_no_DPD)
        fft_no_dpd = np.fft.fft(output_data_no_DPD)
        psd_no_dpd = 10 * np.log10(np.abs(fft_no_dpd) ** 2 / N_no_dpd + 1e-20)
        freq_no_dpd = np.fft.fftfreq(N_no_dpd, d=1/fs)
        
        # Shift to center
        psd_dpd_shifted = np.fft.fftshift(psd_dpd)
        freq_dpd_shifted = np.fft.fftshift(freq_dpd)
        
        psd_no_dpd_shifted = np.fft.fftshift(psd_no_dpd)
        freq_no_dpd_shifted = np.fft.fftshift(freq_no_dpd)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(freq_dpd_shifted, psd_dpd_shifted, label='With DPD', alpha=0.7)
        plt.plot(freq_no_dpd_shifted, psd_no_dpd_shifted, label='Without DPD', alpha=0.7)
        
        # Mark channel boundaries
        y_min, y_max = plt.ylim()
        
        # Main channel
        plt.axvline(-channel_bw/2, color='g', linestyle='--', alpha=0.5, label='Main Channel')
        plt.axvline(channel_bw/2, color='g', linestyle='--', alpha=0.5)
        
        # Lower adjacent channel
        plt.axvline(-channel_spacing - channel_bw/2, color='r', linestyle='--', alpha=0.5, label='Adjacent Channels')
        plt.axvline(-channel_spacing + channel_bw/2, color='r', linestyle='--', alpha=0.5)
        
        # Upper adjacent channel
        plt.axvline(channel_spacing - channel_bw/2, color='r', linestyle='--', alpha=0.5)
        plt.axvline(channel_spacing + channel_bw/2, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Normalized Frequency (cycles/sample)')
        plt.ylabel('Power Spectral Density (dB)')
        plt.title(f'Spectrum with ACLR Channels\n'
                  f'ACLR Lower (DPD): {aclr_metrics["aclr_lower_dpd"]:.2f} dBc, '
                  f'Upper (DPD): {aclr_metrics["aclr_upper_dpd"]:.2f} dBc\n'
                  f'ACLR Lower (No DPD): {aclr_metrics["aclr_lower_no_dpd"]:.2f} dBc, '
                  f'Upper (No DPD): {aclr_metrics["aclr_upper_no_dpd"]:.2f} dBc')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_signals(dataset):
        """Plot input magnitude vs output magnitude"""
        # include the NMSE on the plot, along wiht the line y=x
        nmse = dataset.calculate_nmse()
        plt.figure()
        plt.plot([0, max(abs(dataset.input_data))], [0, max(abs(dataset.input_data))], 'r--', label='y=x')
        plt.xlabel('Input Magnitude')
        plt.ylabel('Output Magnitude')
        plt.title(f'Input vs Output Magnitude with NMSE: {nmse:.2f} dB')
        plt.legend()
        
        plt.plot(abs(dataset.input_data), abs(dataset.output_data), '.')
        plt.xlabel('Input Magnitude')
        plt.ylabel('Output Magnitude')
        plt.title('Input vs Output Magnitude')
        plt.grid()
        plt.show()