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
    
    def train_using_ila(self, training_dataset, valid_dataset, iterations, retrain_epochs_per_iteration, seq_length):
        # Block-wise training: each iteration uses a block of size seq_length
        # When all blocks are exhausted, cycle back to the start
        
        total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
        total_samples = len(training_dataset.input_data)
        num_blocks = max(1, (total_samples - seq_length) // seq_length + 1)
        
        # Track which block to use for each iteration (cycle through blocks)
        for iteration in range(iterations):
            # Determine which block to use (cycle back to start when exhausted)
            block_idx = iteration % num_blocks
            start_idx = block_idx * seq_length
            end_idx = min(start_idx + seq_length, total_samples)
            
            # Extract block of training data
            block_input = training_dataset.input_data[start_idx:end_idx]
            block_output = training_dataset.output_data[start_idx:end_idx]
            block_dataset = Dataset(input_data=block_input, output_data=block_output)
            
            # Generate outputs for this block
            inverse_model_output = self.inverse_model.generate_model_output(block_input)
            forward_model_output = self.forward_model.generate_model_output(inverse_model_output)
            
            # Retrain inverse model on the error for this block
            # Create dataset with forward model output vs original input (aligned)
            aligned_input = inverse_model_output[self._get_model_trim_amount(self.forward_model):]
            new_dataset = Dataset(input_data=aligned_input, output_data=forward_model_output)
            train_losses_inv, valid_losses_inv, best_epoch_inv = self.inverse_model.get_best_model(
                num_epochs=retrain_epochs_per_iteration, 
                training_dataset=new_dataset,
                validation_dataset=valid_dataset
            )

            # Calculate NMSE on full training set after this iteration
            full_inverse_output = self.inverse_model.generate_model_output(training_dataset.input_data)
            full_forward_output = self.forward_model.generate_model_output(full_inverse_output)
            dataset = Dataset(training_dataset.input_data[total_trim:], full_forward_output)
            nmse = dataset.calculate_nmse()
            print(f"Iteration {iteration+1}/{iterations} (Block {block_idx+1}/{num_blocks}, samples {start_idx}-{end_idx}) - NMSE: {nmse:.4f} dB")

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
    
    def plot_constellation(self, dataset, fs, nperseg, bw_sub_ch, n_sub_ch, show_dpd_output=True, plot_all_carriers=False):
        """Plot QAM constellation for OFDM signals.
        
        Uses IFFT-frame demodulation for DPA-type datasets (no cyclic prefix).
        Extracts symbols from one or all carriers in the first frame for visualization.
        
        Parameters:
        -----------
        dataset : Dataset
            Dataset with input_data and output_data (PA input and PA output)
        fs : float
            Sampling frequency (Hz)
        nperseg : int
            FFT size / frame length
        bw_sub_ch : float
            Sub-channel bandwidth (Hz)
        n_sub_ch : int
            Number of sub-channels
        show_dpd_output : bool, optional
            If True, show DPD-corrected output. If False, show raw PA output.
            Default is True.
        plot_all_carriers : bool, optional
            If True, plot all carriers on same constellation. If False, plot only one carrier.
            Default is False.
        """
        bin_spacing = fs / nperseg
        n_active = int(round(bw_sub_ch / bin_spacing))
        n_half = n_active // 2
        dc = nperseg // 2
        carrier_f_shifts = [(i - (n_sub_ch - 1) / 2) * bw_sub_ch for i in range(n_sub_ch)]
        carrier_centres = [dc + int(round(f / bin_spacing)) for f in carrier_f_shifts]
        
        def demodulate_single_frame(signal, offset=0, carrier_idx=None):
            """Extract QAM symbols from first complete OFDM frame.
            
            Parameters:
            -----------
            signal : np.ndarray
                Input signal
            offset : int
                Number of samples to skip before extracting frame (for alignment)
            carrier_idx : int or None
                If int, extract from that carrier index. If None, extract from all carriers.
            """
            frame = signal[offset:offset + nperseg]
            fd = np.fft.fftshift(np.fft.fft(frame))
            
            if carrier_idx is not None:
                # Extract from one specific carrier
                cc = carrier_centres[carrier_idx]
                sc_neg = fd[cc - n_half:cc]
                sc_pos = fd[cc + 1:cc + n_half + 1]
                sc = np.concatenate([sc_neg, sc_pos])
                return sc
            else:
                # Extract from all carriers
                all_sc = []
                for cc in carrier_centres:
                    sc_neg = fd[cc - n_half:cc]
                    sc_pos = fd[cc + 1:cc + n_half + 1]
                    sc = np.concatenate([sc_neg, sc_pos])
                    all_sc.append(sc)
                return np.concatenate(all_sc)
        
        # Determine which carrier(s) to extract
        carrier_to_extract = None if plot_all_carriers else 1  # None = all, 1 = second carrier
        
        # Get output constellation first to determine which frame we can compare
        if show_dpd_output:
            # Process through datapath to get DPD-corrected output
            dpd_dataset = self.process(dataset.input_data)
            
            # CRITICAL: Trimming breaks frame alignment!
            # The DPD process trims samples, so we need to extract the SAME logical frame
            # from both input and output for valid comparison.
            total_trim = self._get_model_trim_amount(self.inverse_model) + self._get_model_trim_amount(self.forward_model)
            
            # Find start of next complete frame after trim
            # If trim=30, nperseg=16384, we need offset = 16384-30 = 16354 to reach frame 1
            samples_into_current_frame = total_trim % nperseg
            if samples_into_current_frame == 0:
                offset_output = 0  # Already frame-aligned
                offset_input = 0
            else:
                offset_output = nperseg - samples_into_current_frame  # Skip to next frame in output
                offset_input = offset_output + total_trim  # Same frame in input (accounting for trim)
            
            if len(dpd_dataset.output_data) < offset_output + nperseg:
                print(f"Warning: Not enough data for aligned frame (need {offset_output + nperseg}, have {len(dpd_dataset.output_data)})")
                input_qam = demodulate_single_frame(dataset.input_data, offset=0, carrier_idx=carrier_to_extract)
                output_qam = input_qam  # Fallback
            else:
                # Extract the SAME logical frame from both input and output
                input_qam = demodulate_single_frame(dataset.input_data, offset=offset_input, carrier_idx=carrier_to_extract)
                output_qam = demodulate_single_frame(dpd_dataset.output_data, offset=offset_output, carrier_idx=carrier_to_extract)
            label = 'DPD Output'
            color = 'green'
        else:
            # Use raw PA output (no DPD) - compare frame 0 from both
            input_qam = demodulate_single_frame(dataset.input_data, offset=0, carrier_idx=carrier_to_extract)
            output_qam = demodulate_single_frame(dataset.output_data, offset=0, carrier_idx=carrier_to_extract)
            label = 'PA Output (no DPD)'
            color = 'red'
        
        # Filter out guard bands (low-power subcarrier positions at spectrum edges)
        # The key is to identify which subcarrier INDEX positions have consistently low power,
        # not to filter individual QAM symbols by their power (which would remove center constellation points)
        # We need multiple frames to compute average power per subcarrier position
        
        # Sample several frames from the input to identify guard band positions
        n_frames_available = len(dataset.input_data) // nperseg
        sample_frames = min(10, n_frames_available)  # Sample up to 10 frames
        
        if sample_frames > 1:
            all_powers = []
            for frame_idx in range(sample_frames):
                frame_offset = frame_idx * nperseg
                frame_qam = demodulate_single_frame(dataset.input_data, offset=frame_offset, carrier_idx=carrier_to_extract)
                all_powers.append(np.abs(frame_qam) ** 2)
            
            # Average power per subcarrier index across frames
            avg_power_per_subcarrier = np.mean(all_powers, axis=0)
            median_power = np.median(avg_power_per_subcarrier[avg_power_per_subcarrier > 0])
            
            # Keep subcarrier positions with power > 1% of median
            mask = avg_power_per_subcarrier > 0.01 * median_power
            
            # Apply mask to both input and output
            input_qam_filtered = input_qam[mask]
            output_qam_filtered = output_qam[mask]
            
            carriers_str = f"{n_sub_ch} carriers" if plot_all_carriers else "1 carrier"
            print(f"Total subcarriers ({carriers_str}): {len(input_qam)}, Active (after guard band filtering): {np.sum(mask)}")
        else:
            # Not enough frames for mask - use all subcarriers
            input_qam_filtered = input_qam
            output_qam_filtered = output_qam
            carriers_str = f"{n_sub_ch} carriers" if plot_all_carriers else "1 carrier"
            print(f"Total subcarriers ({carriers_str}): {len(input_qam)} (no filtering, insufficient frames)")
        
        # Normalize output to match input scale (compensate for PA gain)
        input_rms = np.sqrt(np.mean(np.abs(input_qam_filtered) ** 2))
        output_rms = np.sqrt(np.mean(np.abs(output_qam_filtered) ** 2))
        if output_rms > 1e-10:  # Avoid division by zero
            output_qam_scaled = output_qam_filtered * (input_rms / output_rms)
        else:
            output_qam_scaled = output_qam_filtered
        
        # Calculate EVM (Error Vector Magnitude)
        # First, estimate and remove any residual phase/gain offset between frames
        # This handles the case where different OFDM frames have different absolute phases
        # Find optimal complex scaling α that minimizes |input - α*output|²
        scaling_factor = np.sum(np.conj(output_qam_scaled) * input_qam_filtered) / np.sum(np.abs(output_qam_scaled) ** 2)
        output_qam_aligned = output_qam_scaled * scaling_factor
        
        # Now compute EVM after alignment
        error_vector = output_qam_aligned - input_qam_filtered
        evm_rms = np.sqrt(np.mean(np.abs(error_vector) ** 2))
        reference_rms = np.sqrt(np.mean(np.abs(input_qam_filtered) ** 2))
        evm_percent = (evm_rms / reference_rms) * 100
        evm_db = 20 * np.log10(evm_rms / reference_rms)
        
        # Plot both constellations (use aligned version for visual comparison)
        plt.figure(figsize=(10, 10))
        plt.scatter(input_qam_filtered.real, input_qam_filtered.imag, alpha=0.5, s=5, c='blue', label='Input (Ideal)')
        plt.scatter(output_qam_aligned.real, output_qam_aligned.imag, alpha=0.5, s=5, c=color, label=label)
        plt.xlabel('In-Phase (I)', fontsize=12)
        plt.ylabel('Quadrature (Q)', fontsize=12)
        
        carrier_desc = f"All {n_sub_ch} Carriers" if plot_all_carriers else "Single Carrier"
        plt.title(f'QAM Constellation - Single Frame, {carrier_desc}\n({len(input_qam_filtered):,} symbols, guard bands filtered)\nEVM: {evm_percent:.2f}% ({evm_db:.2f} dB)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # Print EVM metrics
        print(f"EVM (RMS): {evm_percent:.3f}%")
        print(f"EVM (dB): {evm_db:.3f} dB")
    
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