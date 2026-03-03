from .Experiment import Experiment
import torch.nn as nn

class LotteryTicketExperiment(Experiment):
    """Similar to regular pruning but we reset the weights to their original position every time we prune"""
    def __init__(self, nn_model, num_prune_iterations, prune_amount, retrain_epochs, training_dataset, valid_dataset, test_dataset, use_frames=True, frame_stride=100, frame_length=500):
        super().__init__(nn_model, training_dataset, valid_dataset, test_dataset)
        self.num_prune_iterations = num_prune_iterations
        self.prune_amount = prune_amount
        self.retrain_epochs = retrain_epochs
        self.use_frames = use_frames
        self.frame_stride = frame_stride
        self.frame_length = frame_length

    def prune(self):
        nmse_results = []
        prune_percentages = []
        valid_losses_final = []
        all_valid_losses = []
        all_best_epochs = []

        linear_layer_names = [
            name for name, module in self.nn_model_copy.nn_model.named_modules()
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
            self.nn_model_copy.prune_model(linear_layer_names, self.prune_amount)
            
            # Calculate current pruning percentage
            current_prune_pct = self.nn_model_copy._get_pruning_percentage()
            prune_percentages.append(current_prune_pct)
            print(f"Current pruning: {current_prune_pct:.2f}% of weights are zero")
            
            # Retrain the model
            print(f"Retraining for {self.retrain_epochs} epochs...")
            train_losses, valid_losses, best_epoch = self.nn_model_copy.get_best_model(
                num_epochs=self.retrain_epochs,
                training_dataset = self.training_dataset,
                validation_dataset = self.valid_dataset,
                use_frames = self.use_frames,
                frame_stride = self.frame_stride,
                frame_length = self.frame_length
            )
            
            # Store validation losses and best epoch
            all_valid_losses.append(valid_losses)
            all_best_epochs.append(best_epoch)
            valid_losses_final.append(min(valid_losses))
            
            # Calculate NMSE
            nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
            nmse_results.append(nmse)
            print(f"NMSE: {nmse:.4f} dB")
        
        return prune_percentages, nmse_results, valid_losses_final, all_best_epochs, all_valid_losses