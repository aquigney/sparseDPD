import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import numpy as np

class Experiment:
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
        prune_percentages, nmse_results, valid_losses_final, all_best_epochs, all_valid_losses =  self.prune()
        self.plot_results(prune_percentages, nmse_results, initial_nmse)
        self.plot_training_curves(all_valid_losses, all_best_epochs, prune_percentages)


    def plot_results(self, prune_percentages, nmse_results, initial_nmse):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Add the initial values to the results
        nmse_to_plot = [initial_nmse] + nmse_results
        prune_pct_to_plot = [0] + prune_percentages
        
        # Plot NMSE
        ax.plot(prune_pct_to_plot, nmse_to_plot, marker='o', linewidth=2, markersize=8, color='tab:blue')
        ax.set_xlabel('Pruning Percentage (%)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title('Model Performance vs Pruning Percentage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_training_curves(self, all_valid_losses, all_best_epochs, prune_percentages):
        

        n_iterations = len(all_valid_losses)
        colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))

        fig, ax = plt.subplots(figsize=(12, 6))

        epoch_offset = 0
        prev_end_epoch = None
        prev_end_loss = None

        for i, (valid_losses, best_epoch, prune_pct, color) in enumerate(
            zip(all_valid_losses, all_best_epochs, prune_percentages, colors)
        ):
            n_epochs = len(valid_losses)
            epochs = np.arange(1, n_epochs + 1) + epoch_offset

            label = f'Iter {i+1}: {prune_pct:.1f}% pruned'

            # Plot validation loss curve
            ax.plot(epochs, valid_losses, linewidth=2, color=color, label=label)

            # Draw connecting line from previous iteration
            if prev_end_epoch is not None:
                ax.plot(
                    [prev_end_epoch, epochs[0]],
                    [prev_end_loss, valid_losses[0]],
                    color=color,
                    linewidth=2,
                    alpha=0.6
                )

            # Best epoch marker
            best_epoch_global = epoch_offset + best_epoch
            best_loss = valid_losses[best_epoch - 1]

            ax.axvline(
                x=best_epoch_global,
                color=color,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )

            ax.plot(
                best_epoch_global,
                best_loss,
                marker='*',
                markersize=12,
                color=color,
                markeredgecolor='black',
                markeredgewidth=1
            )

            # Store end of this iteration for next connection
            prev_end_epoch = epochs[-1]
            prev_end_loss = valid_losses[-1]

            epoch_offset += n_epochs

        ax.set_xlabel('Epoch (cumulative)', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(
            'Validation Loss Curves (Sequential with Pruning Transitions)',
            fontsize=14,
            fontweight='bold'
        )

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
