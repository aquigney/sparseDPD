# Create a Dataset object for loading and processing data
import numpy as np
import scipy
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def input_phase(self):
        Ax = np.abs(self.input_data)
        return np.conj(self.input_data)/Ax

    def calculate_nmse(self):
        power_input = np.mean(np.abs(self.input_data)**2)
        power_error = np.mean(np.abs(self.input_data - self.output_data)**2)
        nmse = 10 * np.log10(power_error / power_input)
        return nmse

    @staticmethod
    def conj_phase(signal):
        x = np.asarray(signal)
        return np.exp(-1j * np.angle(x))
    
    def plot_spectrum(self, title='Spectrum', show_plot=True, 
                     fs=1.0, freq_unit='Normalized', figsize=(10, 6), num_points=1024, save_path=None, plot_input=True, plot_output=True):
        # Create figure with high-quality settings for reports
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Compute FFT and power spectral density for input signal
        input_data = self.input_data[:num_points]  # Use only the first num_points for FFT
        n_input = len(input_data)
        freq_input = np.fft.fftshift(np.fft.fftfreq(n_input, d=1/fs))
        fft_input = np.fft.fftshift(np.fft.fft(input_data))
        psd_input = 20 * np.log10(np.abs(fft_input) / n_input + 1e-12)  # Add small value to avoid log(0)
        
        # Compute FFT and power spectral density for output signal
        output_data = self.output_data[:num_points]  # Use only the first num_points for FFT
        n_output = len(output_data)
        freq_output = np.fft.fftshift(np.fft.fftfreq(n_output, d=1/fs))
        fft_output = np.fft.fftshift(np.fft.fft(output_data))
        psd_output = 20 * np.log10(np.abs(fft_output) / n_output + 1e-12)
        
        # Normalize PSD (set peak to 0 dB)
        psd_input_norm = psd_input - np.max(psd_input)
        psd_output_norm = psd_output - np.max(psd_output)
        
        # Plot the spectra (output first, then input so input appears on top)
        if plot_output:
            ax.plot(freq_output, psd_output_norm, label='Output', linewidth=1.5, alpha=0.9, color='#ff7f0e')
        if plot_input:
            ax.plot(freq_input, psd_input_norm, label='Input', linewidth=1.5, alpha=0.9, color='#1f77b4')
        
        # Formatting for publication quality
        ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        
        # Set y-axis limits for better visualization
        ax.set_ylim([-80, 5])
        
        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def plot_constellation(self, fs, nperseg, bw_sub_ch, n_sub_ch, plot_all_carriers=False):
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
                # Extract from single carrier
                center = carrier_centres[carrier_idx]
                qam_symbols = fd[center - n_half:center + n_half]
                return qam_symbols
            else:
                # Extract from all carriers
                all_symbols = []
                for center in carrier_centres:
                    qam_symbols = fd[center - n_half:center + n_half]
                    all_symbols.extend(qam_symbols)
                return np.array(all_symbols)
        
        # Determine which carrier(s) to extract
        carrier_to_extract = None if plot_all_carriers else 1  # None = all, 1 = second carrier
        
        # Demodulate input and output from the same frame (frame 0)
        input_qam = demodulate_single_frame(self.input_data, offset=0, carrier_idx=carrier_to_extract)
        output_qam = demodulate_single_frame(self.output_data, offset=0, carrier_idx=carrier_to_extract)
        
        # Filter out guard bands (low-power subcarrier positions at spectrum edges)
        # Sample several frames from the input to identify guard band positions
        n_frames_available = len(self.input_data) // nperseg
        sample_frames = min(10, n_frames_available)  # Sample up to 10 frames
        
        if sample_frames > 1:
            all_powers = []
            for frame_idx in range(sample_frames):
                frame_qam = demodulate_single_frame(self.input_data, offset=frame_idx * nperseg, carrier_idx=carrier_to_extract)
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
        # Estimate and remove any residual phase/gain offset between frames
        # Find optimal complex scaling α that minimizes |input - α*output|²
        scaling_factor = np.sum(np.conj(output_qam_scaled) * input_qam_filtered) / np.sum(np.abs(output_qam_scaled) ** 2)
        output_qam_aligned = output_qam_scaled * scaling_factor
        
        # Compute EVM after alignment
        error_vector = output_qam_aligned - input_qam_filtered
        evm_rms = np.sqrt(np.mean(np.abs(error_vector) ** 2))
        reference_rms = np.sqrt(np.mean(np.abs(input_qam_filtered) ** 2))
        evm_percent = (evm_rms / reference_rms) * 100
        evm_db = 20 * np.log10(evm_rms / reference_rms)
        
        # Plot both constellations
        plt.figure(figsize=(10, 10))
        plt.scatter(input_qam_filtered.real, input_qam_filtered.imag, alpha=0.5, s=5, c='blue', label='Input (Ideal)')
        plt.scatter(output_qam_aligned.real, output_qam_aligned.imag, alpha=0.5, s=5, c='red', label='PA Output')
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
        
        return evm_percent, evm_db
    
    