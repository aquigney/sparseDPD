# Datapath extension to work with a physical PA

from sparseDPD import Datapath, Dataset, NeuralNetwork, Volterra


class PhysicalDatapath(Datapath):
    def __init__(self, inverse_model):
        super().__init__(forward_model=None, inverse_model=inverse_model)
        # Initialize the physical PA here (e.g., set up serial communication, configure settings, etc.)
        self.initialize_physical_pa()
    
    def initialize_physical_pa(self):
        pass

    def process(self, input_signal):
        # Process input signal through inverse_model and send to physical PA.
        if type(self.inverse_model) == Volterra:
            pre_distorted_signal = self.inverse_model.build_y(input_signal)
        elif isinstance(self.inverse_model, NeuralNetwork):
            pre_distorted_signal = self.inverse_model.generate_model_output(input_signal)
        else: 
            print(f"Your inverse model is type {type(self.inverse_model)} and your forward model is type {type(self.forward_model)}")
            # give an error
            raise TypeError("Unsupported model types for PhysicalDatapath")

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
    

    def train_using_ila(self, training_dataset, valid_dataset, iterations, retrain_epochs_per_iteration, seq_length=None):
        # Block-wise training: each iteration uses a block of size seq_length
        # When all blocks are exhausted, cycle back to the start
        if seq_length == None:
            seq_length = len(training_dataset.input_data)
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
            forward_model_output = self.send_to_physical_pa(inverse_model_output)
            
            # Retrain inverse model on the error for this block
            # Create dataset with forward model output vs original input (aligned)
            aligned_input = inverse_model_output[self._get_model_trim_amount(self.forward_model):]
            new_dataset = Dataset(input_data=aligned_input, output_data=forward_model_output)
            train_losses_inv, valid_losses_inv, best_epoch_inv = self.inverse_model.get_best_model(
                num_epochs=retrain_epochs_per_iteration, 
                training_dataset=new_dataset,
                validation_dataset=valid_dataset
            )

            dataset = Dataset(input_data=training_dataset.input_data[total_trim:], output_data=forward_model_output)
            nmse = dataset.calculate_nmse()
            print(f"Iteration {iteration+1}/{iterations} (Block {block_idx+1}/{num_blocks}, samples {start_idx}-{end_idx}) - NMSE: {nmse:.4f} dB")
            print("-"*50)