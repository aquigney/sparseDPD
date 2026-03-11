import copy

import torch.nn as nn

from .Experiment import Experiment


class AdaptivePruningExperiment(Experiment):
    """
    Adaptive binary-search linear (weight-level) pruning for PNTDNN.

    Algorithm
    ---------
    1. Record baseline NMSE.
    2. Prune ``initial_prune_fraction`` of weights (L1 global unstructured,
       same method as LinearExperiment), then retrain for ``retrain_epochs``.
    3. If NMSE stays within ``nmse_tolerance`` dB of baseline → accept, reset
       fraction to ``initial_prune_fraction``, repeat on the sparser model.
    4. If NMSE degrades too much → reject, restore model, halve the fraction.
    5. Stop when the fraction drops below ``min_prune_fraction``.

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
        e.g. 0.5 means the pruned model may be at most 0.5 dB worse.
    initial_prune_fraction : float, optional
        Fraction of weights to attempt removing each step. Default 0.5 (50%).
    min_prune_fraction : float, optional
        Stop when the fraction would drop below this value. Default 0.01 (1%).
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
    ):
        super().__init__(nn_model, training_dataset, valid_dataset, test_dataset)
        self.retrain_epochs = retrain_epochs
        self.nmse_tolerance = nmse_tolerance
        self.initial_prune_fraction = initial_prune_fraction
        self.min_prune_fraction = min_prune_fraction

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

        initial_nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
        nmse_threshold = initial_nmse + self.nmse_tolerance

        print(f"Baseline NMSE          : {initial_nmse:.4f} dB")
        print(f"Tolerance              : {self.nmse_tolerance:+.2f} dB")
        print(f"NMSE must stay below   : {nmse_threshold:.4f} dB")
        print(f"Initial prune fraction : {self.initial_prune_fraction * 100:.1f}%")
        print(f"Min prune fraction     : {self.min_prune_fraction * 100:.1f}%")

        step = 0
        current_fraction = self.initial_prune_fraction

        while current_fraction >= self.min_prune_fraction:
            step += 1
            print(f"\n{'='*60}")
            print(f"Step {step}  |  fraction = {current_fraction * 100:.2f}%")
            print(f"{'='*60}")

            # Save model so we can roll back if this step is rejected
            saved_model = copy.deepcopy(self.nn_model_copy)

            # Apply L1 global unstructured pruning (same as LinearExperiment)
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
            print(f"NMSE: {nmse:.4f} dB  (threshold: {nmse_threshold:.4f} dB)")

            if nmse <= nmse_threshold:
                print(f"  ACCEPTED  (delta: {nmse - initial_nmse:+.4f} dB)")
                prune_percentages.append(current_prune_pct)
                nmse_results.append(nmse)
                all_valid_losses.append(valid_losses)
                all_best_epochs.append(best_epoch)
                valid_losses_final.append(min(valid_losses))
                # Try the same aggressive fraction again on the now-sparser model
                current_fraction = self.initial_prune_fraction
            else:
                print(f"  REJECTED  (delta: {nmse - initial_nmse:+.4f} dB) — restoring model")
                self.nn_model_copy = saved_model
                current_fraction /= 2
                print(f"  Reduced fraction to {current_fraction * 100:.2f}%")

        # Final summary
        final_nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
        final_prune_pct = self.nn_model_copy._get_pruning_percentage()

        print(f"\n{'='*60}")
        print(f"Adaptive pruning finished  ({len(nmse_results)} accepted step(s))")
        print(f"Final NMSE     : {final_nmse:.4f} dB  (baseline: {initial_nmse:.4f} dB, delta: {final_nmse - initial_nmse:+.4f} dB)")
        print(f"Final sparsity : {final_prune_pct:.2f}% of weights zeroed")
        print(f"{'='*60}")

        return prune_percentages, nmse_results, valid_losses_final, all_best_epochs, all_valid_losses
