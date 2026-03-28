# Iterative Indirect Learning for DPD Training
import numpy as np
from .Dataset import Dataset
from .Volterra import Volterra
from .NeuralNetwork import NeuralNetwork


class IterativeTrainer:
    """
    Implements iterative indirect learning for Digital Predistortion (DPD).
    
    The algorithm iteratively improves a DPD model by:
    1. Using current DPD to generate synthetic training data
    2. Retraining the DPD on this data
    3. Repeating until convergence
    """
    
    def __init__(self, forward_model, inverse_model, max_iterations=5, convergence_threshold=0.5):
        """
        Initialize iterative trainer.
        
        Args:
            forward_model: PA model (Volterra or NeuralNetwork) - must be trained
            inverse_model: Initial DPD model (Volterra or NeuralNetwork)
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if NMSE improvement < this (in dB)
        """
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.iteration_history = []
        
    def generate_synthetic_dpd_data(self, clean_inputs):
        """
        Generate synthetic DPD training data.
        
        Args:
            clean_inputs: Clean input signals (complex array)
            
        Returns:
            Dataset with (input=clean_signal, output=predistorted_signal)
        """
        # Step 1: Apply current DPD model to clean inputs
        if isinstance(self.inverse_model, Volterra):
            predistorted = self.inverse_model.build_y(clean_inputs)
        else:  # NeuralNetwork
            predistorted = self.inverse_model.generate_model_output(clean_inputs)
        
        # Step 2: Pass predistorted signals through PA
        if isinstance(self.forward_model, Volterra):
            pa_output = self.forward_model.build_y(predistorted)
        else:  # NeuralNetwork
            pa_output = self.forward_model.generate_model_output(predistorted)
        
        # Step 3: Align signals - we want DPD to learn: clean_input → predistorted_signal
        # such that PA(predistorted) ≈ clean_input
        
        # Calculate total memory offset
        inv_mem = self.inverse_model.num_memory_levels
        fwd_mem = self.forward_model.num_memory_levels
        total_offset = inv_mem + fwd_mem
        
        # Align the signals
        aligned_clean_input = clean_inputs[total_offset:]
        aligned_predistorted = predistorted[fwd_mem:]
        
        # Ensure same length
        min_len = min(len(aligned_clean_input), len(aligned_predistorted))
        
        # Create dataset: input = clean signal, output = what the DPD should produce
        synthetic_dataset = Dataset(
            aligned_clean_input[:min_len],
            aligned_predistorted[:min_len]
        )
        
        return synthetic_dataset, pa_output
    
    def calculate_linearization_nmse(self, test_inputs):
        """
        Calculate how well the full DPD+PA chain linearizes the system.
        
        Args:
            test_inputs: Clean test signals
            
        Returns:
            NMSE in dB (lower is better)
        """
        # Apply DPD
        if isinstance(self.inverse_model, Volterra):
            predistorted = self.inverse_model.build_y(test_inputs)
        else:
            predistorted = self.inverse_model.generate_model_output(test_inputs)
        
        # Apply PA
        if isinstance(self.forward_model, Volterra):
            pa_output = self.forward_model.build_y(predistorted)
        else:
            pa_output = self.forward_model.generate_model_output(predistorted)
        
        # Align signals
        inv_mem = self.inverse_model.num_memory_levels
        fwd_mem = self.forward_model.num_memory_levels
        total_offset = inv_mem + fwd_mem
        
        aligned_input = test_inputs[total_offset:]
        
        # Ensure same length
        min_len = min(len(aligned_input), len(pa_output))
        aligned_input = aligned_input[:min_len]
        pa_output = pa_output[:min_len]
        
        # Calculate NMSE
        error_power = np.mean(np.abs(aligned_input - pa_output)**2)
        signal_power = np.mean(np.abs(aligned_input)**2)
        nmse_db = 10 * np.log10(error_power / signal_power)
        
        return nmse_db
    
    def train_iteration(self, training_dataset, validation_dataset, **train_kwargs):
        """
        Perform one iteration of training.
        
        Args:
            training_dataset: Dataset with clean input/output pairs
            validation_dataset: Validation dataset
            **train_kwargs: Additional arguments for model training
            
        Returns:
            dict with iteration results
        """
        # Generate synthetic DPD training data
        synthetic_train_data, _ = self.generate_synthetic_dpd_data(training_dataset.input_data)
        synthetic_valid_data, _ = self.generate_synthetic_dpd_data(validation_dataset.input_data)
        
        # Retrain DPD model on synthetic data
        if isinstance(self.inverse_model, Volterra):
            # Volterra: retrain with new data
            self.inverse_model.retrain(synthetic_train_data)
        else:  # NeuralNetwork
            # NN: train for more epochs
            num_epochs = train_kwargs.get('num_epochs', 200)
            learning_rate = train_kwargs.get('learning_rate', 1e-3)
            
            self.inverse_model.get_best_model(
                num_epochs=num_epochs,
                training_dataset=synthetic_train_data,
                validation_dataset=synthetic_valid_data,
                learning_rate=learning_rate
            )
        
        # Calculate linearization performance
        train_nmse = self.calculate_linearization_nmse(training_dataset.input_data)
        valid_nmse = self.calculate_linearization_nmse(validation_dataset.input_data)
        
        return {
            'train_nmse': train_nmse,
            'valid_nmse': valid_nmse,
            'synthetic_train_size': len(synthetic_train_data.input_data),
            'synthetic_valid_size': len(synthetic_valid_data.input_data)
        }
    
    def train(self, training_dataset, validation_dataset, **train_kwargs):
        """
        Run iterative indirect learning.
        
        Args:
            training_dataset: Dataset with clean PA input/output pairs
            validation_dataset: Validation dataset
            **train_kwargs: Additional arguments for model training (e.g., num_epochs, learning_rate)
            
        Returns:
            dict with final results and iteration history
        """
        print("="*70)
        print("Starting Iterative Indirect Learning for DPD")
        print("="*70)
        
        # Calculate initial performance
        initial_nmse = self.calculate_linearization_nmse(validation_dataset.input_data)
        print(f"Iteration 0 (Initial): Validation NMSE = {initial_nmse:.2f} dB")
        
        self.iteration_history = [{
            'iteration': 0,
            'valid_nmse': initial_nmse,
            'improvement': 0.0
        }]
        
        previous_nmse = initial_nmse
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\nIteration {iteration}/{self.max_iterations}")
            print("-" * 70)
            
            # Perform one iteration
            results = self.train_iteration(training_dataset, validation_dataset, **train_kwargs)
            
            # Calculate improvement
            improvement = previous_nmse - results['valid_nmse']
            
            # Store results
            iter_results = {
                'iteration': iteration,
                'train_nmse': results['train_nmse'],
                'valid_nmse': results['valid_nmse'],
                'improvement': improvement
            }
            self.iteration_history.append(iter_results)
            
            # Print status
            print(f"  Training NMSE:   {results['train_nmse']:7.2f} dB")
            print(f"  Validation NMSE: {results['valid_nmse']:7.2f} dB")
            print(f"  Improvement:     {improvement:7.2f} dB")
            
            # Check convergence
            if improvement < self.convergence_threshold and improvement >= 0:
                print(f"\nConverged! Improvement ({improvement:.2f} dB) < threshold ({self.convergence_threshold:.2f} dB)")
                break
            
            if improvement < 0:
                print(f"\nWarning: Performance degraded by {-improvement:.2f} dB")
            
            previous_nmse = results['valid_nmse']
        
        print("\n" + "="*70)
        print("Iterative Training Complete")
        print("="*70)
        print(f"Initial NMSE:  {initial_nmse:7.2f} dB")
        print(f"Final NMSE:    {results['valid_nmse']:7.2f} dB")
        print(f"Total Improvement: {initial_nmse - results['valid_nmse']:7.2f} dB")
        print("="*70)
        
        return {
            'final_train_nmse': results['train_nmse'],
            'final_valid_nmse': results['valid_nmse'],
            'total_improvement': initial_nmse - results['valid_nmse'],
            'iterations_completed': len(self.iteration_history) - 1,
            'history': self.iteration_history
        }
    
    def plot_convergence(self):
        """Plot NMSE vs iteration to visualize convergence."""
        import matplotlib.pyplot as plt
        
        iterations = [h['iteration'] for h in self.iteration_history]
        nmse_values = [h['valid_nmse'] for h in self.iteration_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, nmse_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Validation NMSE (dB)', fontsize=12)
        plt.title('Iterative Indirect Learning Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()