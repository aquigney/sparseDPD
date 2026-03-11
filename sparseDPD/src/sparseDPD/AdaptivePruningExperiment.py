import copy

import torch.nn as nn

from .Experiment import Experiment


class AdaptivePruningExperiment(Experiment):
    """
    Adaptive binary-search linear (weight-level) pruning for PNTDNN.

    This version adjusts the pruning step size based on the observed NMSE
    headroom after each accepted step:

    - If pruning has very little effect on NMSE, increase the next prune fraction
    - If pruning is acceptable and near the limit, decrease the next prune fraction
    - If pruning is rejected, restore the model and halve the prune fraction

    Parameters
    ----------
    nn_model : NeuralNetwork
        A fully-trained model (e.g. PNTDNN_NeuralNetwork).
    retrain_epochs : int
        Maximum fine-tuning epochs after each pruning step.
    training_dataset, valid_dataset, test_dataset : Dataset
        The three data splits.
    nmse_tolerance : float
        Maximum allowed NMSE increase in dB relative to baseline.
    initial_prune_fraction : float, optional
        Initial fraction of remaining weights to attempt removing each step.
    min_prune_fraction : float, optional
        Stop when the fraction would drop below this value.
    max_prune_fraction : float, optional
        Maximum fraction of remaining weights to prune in one step.
    """

    def __init__(
        self,
        nn_model,
        retrain_epochs,
        training_dataset,
        valid_dataset,
        test_dataset,
        nmse_tolerance,
        initial_prune_fraction=0.5,
        min_prune_fraction=0.01,
        max_prune_fraction=0.5,
    ):
        super().__init__(nn_model, training_dataset, valid_dataset, test_dataset)
        self.retrain_epochs = retrain_epochs
        self.nmse_tolerance = nmse_tolerance
        self.initial_prune_fraction = initial_prune_fraction
        self.min_prune_fraction = min_prune_fraction
        self.max_prune_fraction = max_prune_fraction

    def prune(self):
        nmse_results = []
        prune_percentages = []
        valid_losses_final = []
        all_valid_losses = []
        all_best_epochs = []
        attempted_fractions = []

        linear_layer_names = [
            name for name, module in self.nn_model_copy.nn_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layer_names:
            raise ValueError("No linear layers found to prune in model")

        initial_nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
        nmse_threshold = initial_nmse + self.nmse_tolerance

        print(f"Baseline NMSE          : {initial_nmse:.4f} dB")
        print(f"Tolerance              : {self.nmse_tolerance:+.2f} dB")
        print(f"NMSE must stay below   : {nmse_threshold:.4f} dB")
        print(f"Initial prune fraction : {self.initial_prune_fraction * 100:.1f}%")
        print(f"Min prune fraction     : {self.min_prune_fraction * 100:.1f}%")
        print(f"Max prune fraction     : {self.max_prune_fraction * 100:.1f}%")

        step = 0
        current_fraction = self.initial_prune_fraction

        while current_fraction >= self.min_prune_fraction:
            step += 1
            print(f"\n{'='*60}")
            print(f"Step {step}  |  fraction = {current_fraction * 100:.2f}%")
            print(f"{'='*60}")

            attempted_fractions.append(current_fraction)

            # Save model so we can roll back if this step is rejected
            saved_model = copy.deepcopy(self.nn_model_copy)

            # Apply L1 global unstructured pruning
            print(f"Pruning {current_fraction * 100:.2f}% of remaining weights...")
            self.nn_model_copy.prune_model(linear_layer_names, current_fraction)

            current_prune_pct = self.nn_model_copy._get_pruning_percentage()
            print(f"Current sparsity: {current_prune_pct:.2f}% of weights are zero")

            # Retrain
            print(f"Retraining for {self.retrain_epochs} epochs...")
            train_losses, valid_losses, best_epoch = self.nn_model_copy.get_best_model(
                num_epochs=self.retrain_epochs,
                training_dataset=self.training_dataset,
                validation_dataset=self.valid_dataset,
            )

            nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
            delta_nmse = nmse - initial_nmse
            print(f"NMSE: {nmse:.4f} dB  (threshold: {nmse_threshold:.4f} dB)")

            if nmse <= nmse_threshold:
                print(f"  ACCEPTED  (delta: {delta_nmse:+.4f} dB)")
                prune_percentages.append(current_prune_pct)
                nmse_results.append(nmse)
                all_valid_losses.append(valid_losses)
                all_best_epochs.append(best_epoch)
                valid_losses_final.append(min(valid_losses))

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
                self.nn_model_copy = saved_model
                current_fraction = max(self.min_prune_fraction, 0.5 * current_fraction)
                print(f"  Reduced fraction to {current_fraction * 100:.2f}%")

        # Final summary
        final_nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
        final_prune_pct = self.nn_model_copy._get_pruning_percentage()

        print(f"\n{'='*60}")
        print(f"Adaptive pruning finished  ({len(nmse_results)} accepted step(s))")
        print(
            f"Final NMSE     : {final_nmse:.4f} dB  "
            f"(baseline: {initial_nmse:.4f} dB, delta: {final_nmse - initial_nmse:+.4f} dB)"
        )
        print(f"Final sparsity : {final_prune_pct:.2f}% of weights zeroed")
        print(f"{'='*60}")

        return (
            prune_percentages,
            nmse_results,
            valid_losses_final,
            all_best_epochs,
            all_valid_losses,
        )