import torch
import torch.nn as nn
from torch.nn.utils import prune

from .Experiment import Experiment


class NodePruningExperiment(Experiment):
    def __init__(
        self,
        nn_model,
        num_prune_iterations,
        prune_amount,
        retrain_epochs,
        training_dataset,
        valid_dataset,
        test_dataset,
        use_frames=True,
        frame_stride=100,
        frame_length=500,
    ):
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



        for i in range(self.num_prune_iterations):
            print(f"\n{'='*60}")
            print(f"Pruning Iteration {i+1}/{self.num_prune_iterations}")
            print(f"{'='*60}")

            print(f"Pruning weakest {self.prune_amount*100:.1f}% of hidden nodes...")
            self._prune_weakest_nodes()

            current_prune_pct = self.nn_model_copy._get_pruning_percentage()
            prune_percentages.append(current_prune_pct)
            print(f"Current pruning: {current_prune_pct:.2f}% of weights are zero")

            print(f"Retraining for {self.retrain_epochs} epochs...")
            train_losses, valid_losses, best_epoch = self.nn_model_copy.get_best_model(
                num_epochs=self.retrain_epochs,
                training_dataset=self.training_dataset,
                validation_dataset=self.valid_dataset,
                use_frames=self.use_frames,
                frame_stride=self.frame_stride,
                frame_length=self.frame_length,
                
            )

            all_valid_losses.append(valid_losses)
            all_best_epochs.append(best_epoch)
            valid_losses_final.append(min(valid_losses))

            nmse = self.nn_model_copy.calculate_forward_nmse(self.test_dataset)
            nmse_results.append(nmse)
            print(f"NMSE: {nmse:.4f} dB")

        return prune_percentages, nmse_results, valid_losses_final, all_best_epochs, all_valid_losses

    def _prune_weakest_nodes(self):
        model = self.nn_model_copy.nn_model

        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if len(linear_layers) < 2:
            print("No hidden linear layers found to prune.")
            return

        hidden_layers = linear_layers[:-1]

        candidates = []
        for layer in hidden_layers:
            if not hasattr(layer, "weight_mask"):
                prune.custom_from_mask(layer, name="weight", mask=torch.ones_like(layer.weight))
            if layer.bias is not None and not hasattr(layer, "bias_mask"):
                prune.custom_from_mask(layer, name="bias", mask=torch.ones_like(layer.bias))

            weight_mask = layer.weight_mask.detach().clone()
            bias_mask = layer.bias_mask.detach().clone() if layer.bias is not None else None

            for node_idx in range(layer.out_features):
                row_active = weight_mask[node_idx].sum() > 0
                bias_active = (bias_mask[node_idx] > 0) if bias_mask is not None else False
                if not (row_active or bias_active):
                    continue

                weight_score = (layer.weight.detach()[node_idx].abs() * weight_mask[node_idx]).sum()
                bias_score = 0.0
                if bias_mask is not None:
                    bias_score = (layer.bias.detach()[node_idx].abs() * bias_mask[node_idx]).item()
                node_score = weight_score.item() + float(bias_score)

                candidates.append((node_score, layer, node_idx))

        if not candidates:
            print("All hidden nodes are already pruned.")
            return

        num_to_prune = int(len(candidates) * self.prune_amount)
        if self.prune_amount > 0 and num_to_prune == 0:
            num_to_prune = 1
        num_to_prune = min(num_to_prune, len(candidates))

        weakest = sorted(candidates, key=lambda x: x[0])[:num_to_prune]
        for _, layer, node_idx in weakest:
            layer.weight_mask.data[node_idx, :] = 0
            if layer.bias is not None and hasattr(layer, "bias_mask"):
                layer.bias_mask.data[node_idx] = 0

        print(f"Pruned {num_to_prune} hidden nodes")
