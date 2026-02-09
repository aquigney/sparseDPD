import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import numpy as np

class Experiment:
    """General Experiment class, instantiates well trained original model"""
    def __init__(self, nn_model, training_dataset, valid_dataset, test_dataset):
        self.original_nn_model = nn_model  # Keep original model untouched
        self.nn_model_copy = copy.deepcopy(nn_model)  # Will be set to working copy

        # Dataset
        self.training_dataset = training_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    
    def run(self):
        initial_nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
        
        initial_valid_loss = self.nn_model_copy._calculate_initial_valid_loss(self.valid_dataset)
        prune_percentages, nmse_results, all_valid_losses, all_best_epochs, all_valid_losses =  self.prune()
        self.plot_results(prune_percentages, nmse_results, initial_nmse)
        self.plot_training_curves(all_valid_losses, all_best_epochs, prune_percentages)


    def plot_results(self, prune_percentages, nmse_results, initial_nmse):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Add the initial values to the results
        nmse_results.insert(0, initial_nmse)
        prune_percentages.insert(0, 0)
        
        # Plot NMSE
        ax.plot(prune_percentages, nmse_results, marker='o', linewidth=2, markersize=8, color='tab:blue')
        ax.set_xlabel('Pruning Percentage (%)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title('Model Performance vs Pruning Percentage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_curves(self, all_valid_losses, all_best_epochs, prune_percentages):
        """Plot validation loss curves for each pruning iteration with best epoch markers"""
        n_iterations = len(all_valid_losses)
        colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (valid_losses, best_epoch, prune_pct, color) in enumerate(zip(all_valid_losses, all_best_epochs, prune_percentages, colors)):
            epochs = range(1, len(valid_losses) + 1)
            label = f'Iter {i+1}: {prune_pct:.1f}% pruned'
            
            # Plot validation loss curve
            ax.plot(epochs, valid_losses, linewidth=2, color=color, label=label, alpha=0.7)
            
            # Add vertical line at best epoch
            ax.axvline(x=best_epoch, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Add marker at best epoch
            best_loss = valid_losses[best_epoch - 1]
            ax.plot(best_epoch, best_loss, marker='*', markersize=12, color=color, 
                   markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss Curves for Each Pruning Iteration', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    