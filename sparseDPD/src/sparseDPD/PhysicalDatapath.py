# Datapath extension to work with a physical PA

from sparseDPD import Datapath, NeuralNetwork, Volterra


class PhysicalDatapath(Datapath):
    def __init__(self, inverse_model):
        super().__init__(forward_model=None, inverse_model=inverse_model)
        # Initialize the physical PA here (e.g., set up serial communication, configure settings, etc.)
        self.initialize_physical_pa()

    def process(self, input_signal):
        # Process input signal through inverse_model and send to physical PA.
        if type(self.inverse_model) == Volterra:
            pre_distorted_signal = self.inverse_model.build_y(input_signal)
        elif isinstance(self.inverse_model, NeuralNetwork):
            pre_distorted_signal = self.inverse_model.generate_model_output(input_signal)
        else: 
            print(f"Your inverse model is type {type(self.inverse_model)} and your forward model is type {type(self.forward_model)}")


        output_signal = self.send_to_physical_pa(pre_distorted_signal)
        return output_signal
    

    def send_to_physical_pa(self, signal):
        # Code to send signal to physical PA and receive output
        # For now, just write the signal to a file, and read the output from another file (simulate physical PA processing)
        with open('physical_pa_input.csv', 'w') as f:
            for sample in signal:
                f.write(f"{sample.real},{sample.imag}\n")
        # Simulate physical PA processing by reading from output file (in practice, this would be the actual output from the PA)
        output_signal = []
        with open('physical_pa_output.csv', 'r') as f:
            for line in f:
                real, imag = map(float, line.strip().split(','))
                output_signal.append(complex(real, imag))
        return output_signal