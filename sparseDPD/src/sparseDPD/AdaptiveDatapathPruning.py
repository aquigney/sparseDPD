import copy
from sparseDPD.DatapathPruningExperiment import DatapathPruningExperiment
import torch.nn as nn


class AdaptiveDatapathPruningExperiment(DatapathPruningExperiment):
    def __init__(self, datapath, training_dataset, validation_dataset, test_dataset, 
                 retrain_epochs, ila_iterations=3, 
                 nmse_tolerance=1, initial_prune_fraction=0.1, min_prune_fraction=0.01, max_prune_fraction=0.5):
        
        # No need for num pruning iterations or prune amount
        super().__init__(datapath=datapath, training_dataset=training_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset, num_prune_iterations=None, prune_amount=None, retrain_epochs=retrain_epochs, ila_iterations=ila_iterations, nmse_tolerance=nmse_tolerance)
        
        self.initial_prune_fraction = initial_prune_fraction
        self.min_prune_fraction = min_prune_fraction
        self.max_prune_fraction = max_prune_fraction

    def prune(self):
        original_nmse = self.original_datapath.calculate_nmse(self.test_dataset.input_data)
        nmse_threshold = original_nmse + self.nmse_tolerance
        nmse_results = []
        prune_percentages = []

        linear_layer_names = [
            name for name, module in self.inverse_model_copy.nn_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layer_names:
            raise ValueError("No linear layers found to prune in model")

        step = 0
        current_fraction = self.initial_prune_fraction

        while current_fraction >= self.min_prune_fraction:
            step += 1
            print(f"\n{'='*60}")
            print(f"Step {step}  |  fraction = {current_fraction * 100:.2f}%")
            print(f"{'='*60}")

            # Save model so we can roll back if this step is rejected
            saved_model = copy.deepcopy(self.datapath_copy.inverse_model)

            # Apply L1 global unstructured pruning
            print(f"Pruning {current_fraction * 100:.2f}% of remaining weights...")
            self.datapath_copy.inverse_model.prune_model(linear_layer_names, current_fraction)

            current_prune_pct = self.datapath_copy.inverse_model._get_pruning_percentage()
            print(f"Current sparsity: {current_prune_pct:.2f}% of weights are zero")

            # Retrain
            print(f"Retraining for {self.retrain_epochs} epochs...")
            self.datapath_copy.train_using_ila(
                training_dataset = self.training_dataset,
                valid_dataset = self.validation_dataset,
                iterations = self.ila_iterations,
                retrain_epochs_per_iteration=self.retrain_epochs
            )

            nmse = self.datapath_copy.calculate_nmse(self.test_dataset.input_data)
            delta_nmse = nmse - original_nmse
            print(f"NMSE: {nmse:.4f} dB  (threshold: {nmse_threshold:.4f} dB)")

            if nmse <= nmse_threshold:
                print(f"  ACCEPTED  (delta: {delta_nmse:+.4f} dB)")
                prune_percentages.append(current_prune_pct)
                nmse_results.append(nmse)

                # Headroom-based adaptive step size
                if delta_nmse < self.nmse_tolerance / 3:
                    # Plenty of headroom: be more aggressive next time
                    current_fraction = min(self.max_prune_fraction, 1.5 * current_fraction)
                    print(
                        f"  Plenty of headroom -> increasing next fraction to "
                        f"{current_fraction * 100:.2f}%"
                    )
                elif delta_nmse > 2 * self.nmse_tolerance / 3:
                    # Close to threshold: be more conservative
                    current_fraction = max(self.min_prune_fraction, 0.75 * current_fraction)
                    print(
                        f"  Close to NMSE limit -> reducing next fraction to "
                        f"{current_fraction * 100:.2f}%"
                    )
                else:
                    # Moderate headroom: keep step size the same
                    print(
                        f"  Moderate headroom -> keeping next fraction at "
                        f"{current_fraction * 100:.2f}%"
                    )

            else:
                print(f"  REJECTED  (delta: {delta_nmse:+.4f} dB) — restoring model")
                self.datapath_copy.inverse_model = saved_model
                
                # Check if we can reduce further
                new_fraction = 0.5 * current_fraction
                if new_fraction < self.min_prune_fraction:
                    print(f"  Cannot reduce fraction below minimum ({self.min_prune_fraction * 100:.2f}%) — stopping")
                    break
                    
                current_fraction = new_fraction
                print(f"  Reduced fraction to {current_fraction * 100:.2f}%")

        # Final summary
        final_nmse = self.datapath_copy.calculate_nmse(self.test_dataset.input_data)
        final_prune_pct = self.datapath_copy.inverse_model._get_pruning_percentage()

        print(f"\n{'='*60}")
        print(f"Adaptive pruning finished  ({len(nmse_results)} accepted step(s))")
        print(
            f"Final NMSE     : {final_nmse:.4f} dB  "
            f"(baseline: {original_nmse:.4f} dB, delta: {final_nmse - original_nmse:+.4f} dB)"
        )
        print(f"Final sparsity : {final_prune_pct:.2f}% of weights zeroed")
        print(f"{'='*60}")

        self.best_datapath = copy.deepcopy(self.datapath_copy)  # Final best datapath after adaptive pruning

        return prune_percentages, nmse_results