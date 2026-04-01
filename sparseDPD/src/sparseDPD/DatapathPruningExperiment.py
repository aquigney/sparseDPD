from sparseDPD import Datapath, NeuralNetwork
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

class DatapathPruningExperiment():
    def __init__(self, datapath, training_dataset, validation_dataset, test_dataset,num_prune_iterations, prune_amount, retrain_epochs, ila_iterations=3, nmse_tolerance=1, seq_length=None):
        self.original_datapath = datapath
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.num_prune_iterations = num_prune_iterations
        self.prune_amount = prune_amount
        self.ila_iterations = ila_iterations
        self.retrain_epochs = retrain_epochs
        self.inverse_model_copy = copy.deepcopy(datapath.inverse_model)  # Will be set to working copy
        self.datapath_copy = Datapath(datapath.forward_model, self.inverse_model_copy)  # Create a copy of the datapath with the copied forward model
        self.best_datapath = copy.deepcopy(self.datapath_copy)  # To keep track of the best performing datapath
        self.nmse_tolerance = nmse_tolerance
        if seq_length is None:
            self.seq_length = len(training_dataset.input_data)  # Use full length if not specified
        else:
            self.seq_length = seq_length


    def run(self):
        initial_nmse = self.datapath_copy.calculate_nmse(self.test_dataset.input_data)
        
        prune_percentages, nmse_results, =  self.prune()
        self.plot_results(initial_nmse, nmse_results, prune_percentages)

    def prune(self):
        original_nmse = self.original_datapath.calculate_nmse(self.test_dataset.input_data)
        nmse_results = []
        prune_percentages = []

        linear_layer_names = [
            name for name, module in self.inverse_model_copy.nn_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layer_names:
            raise ValueError("No linear layers found to prune in model")

        for i in range(self.num_prune_iterations):
            print(f"\n{'='*60}")
            print(f"Pruning Iteration {i+1}/{self.num_prune_iterations}")
            print(f"{'='*60}")
            
            # Apply pruning to the current model (iterative pruning of remaining weights)
            print(f"Pruning {self.prune_amount*100:.1f}% of remaining weights...")
            self.inverse_model_copy.prune_model(linear_layer_names, self.prune_amount)
            
            # Calculate current pruning percentage
            current_prune_pct = self.inverse_model_copy._get_pruning_percentage()
            prune_percentages.append(current_prune_pct)
            print(f"Current pruning: {current_prune_pct:.2f}% of weights are zero")
            
            # Retrain the model
            print(f"Retraining for {self.retrain_epochs} epochs...")
            self.datapath_copy.train_using_ila(
                training_dataset = self.training_dataset,
                valid_dataset = self.validation_dataset,
                iterations = self.ila_iterations,
                retrain_epochs_per_iteration=self.retrain_epochs,
                seq_length=self.seq_length
            )
            
            # Calculate NMSE
            nmse = self.datapath_copy.calculate_nmse(self.test_dataset.input_data)
            nmse_results.append(nmse)
            print(f"NMSE: {nmse:.4f} dB")
            
            # Only update if withing NMSE tolerance of original (to avoid over-pruning)
            if nmse < original_nmse + self.nmse_tolerance:
                self.best_datapath = copy.deepcopy(self.datapath_copy)

        return prune_percentages, nmse_results
    
    def plot_results(self, initial_nmse, nmse_results, prune_percentages):
        # Prepend initial values (0% pruning)
        all_prune_pcts = [0] + prune_percentages
        all_nmse = [initial_nmse] + nmse_results
        
        plt.figure(figsize=(10, 6))
        plt.plot(all_prune_pcts, all_nmse, marker='o', label='NMSE vs Pruning')
        plt.title('NMSE vs Pruning Percentage')
        plt.xlabel('Pruning Percentage (%)')
        plt.ylabel('NMSE (dB)')
        plt.grid()
        plt.legend()
        plt.show()
