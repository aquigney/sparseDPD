# Full datapath with forward and inverse models (could be either NN or volterra)


from .Volterra import Volterra
from .Dataset import Dataset
from .NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt


class Datapath:
    def __init__(self, forward_model, inverse_model):
        self.forward_model = forward_model
        self.inverse_model = inverse_model

    def process(self, input_signal):
        """Process the input signal through the inverse model followed by the forward model"""
        if type(self.inverse_model) == Volterra:
            pre_distorted_signal = self.inverse_model.build_y(input_signal)
            # Process through forward model
            if type(self.forward_model) == Volterra:
                output_signal = self.forward_model.build_y(pre_distorted_signal)
                input_signal = input_signal[self.forward_model.num_memory_levels + self.inverse_model.num_memory_levels:]  
        elif type(self.inverse_model) == NeuralNetwork:
            pre_distorted_signal = self.inverse_model.generate_model_output(input_signal)
            if type(self.forward_model) == NeuralNetwork:
                output_signal = self.forward_model.generate_model_output(pre_distorted_signal)
                input_signal = input_signal[self.forward_model.num_memory_levels + self.inverse_model.num_memory_levels:]  # Align input signal
            elif type(self.forward_model) == Volterra:
                output_signal = self.forward_model.build_y(pre_distorted_signal)
                input_signal = input_signal[self.forward_model.num_memory_levels + self.inverse_model.num_memory_levels:]  # Align input signal
        # Trim input signal to line up with output signal and return both 
        dataset = Dataset(input_signal, output_signal)
        return dataset
    
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