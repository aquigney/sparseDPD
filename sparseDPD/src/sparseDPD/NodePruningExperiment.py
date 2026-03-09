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
        learning_rate=1e-4  # Lower default learning rate for fine-tuning after pruning
    ):
        super().__init__(nn_model, training_dataset, valid_dataset, test_dataset)
        self.num_prune_iterations = num_prune_iterations
        self.prune_amount = prune_amount
        self.retrain_epochs = retrain_epochs
        self.learning_rate = learning_rate

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
                validation_dataset=self.valid_dataset
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
        for layer_idx, layer in enumerate(hidden_layers):
            if not hasattr(layer, "weight_mask"):
                prune.custom_from_mask(layer, name="weight", mask=torch.ones_like(layer.weight))
            if layer.bias is not None and not hasattr(layer, "bias_mask"):
                prune.custom_from_mask(layer, name="bias", mask=torch.ones_like(layer.bias))

            weight_mask = layer.weight_mask.detach().clone()
            bias_mask = layer.bias_mask.detach().clone() if layer.bias is not None else None
            
            # Get next layer to check output connections
            next_layer = linear_layers[layer_idx + 1] if layer_idx + 1 < len(linear_layers) else None
            if next_layer is not None and not hasattr(next_layer, "weight_mask"):
                prune.custom_from_mask(next_layer, name="weight", mask=torch.ones_like(next_layer.weight))

            for node_idx in range(layer.out_features):
                # Check if node's output weights are active
                row_active = weight_mask[node_idx].sum() > 0
                if not row_active:
                    continue
                
                # Check if node's output connections to next layer are active
                if next_layer is not None:
                    next_layer_connections = next_layer.weight_mask[:, node_idx].sum()
                    if next_layer_connections == 0:
                        # This node has no active output connections, skip it
                        continue
                
                # Calculate importance based on both magnitude and number of active connections
                # Use L2 norm for better importance estimation
                active_weights = layer.weight.detach()[node_idx] * weight_mask[node_idx]
                weight_score = (active_weights ** 2).sum().sqrt()  # L2 norm
                
                bias_score = 0.0
                if bias_mask is not None and bias_mask[node_idx] > 0:
                    bias_score = layer.bias.detach()[node_idx].abs().item()
                
                # Combine weight and bias importance
                node_score = weight_score.item() + bias_score

                candidates.append((node_score, layer, node_idx))

        if not candidates:
            print("All hidden nodes are already pruned.")
            return

        num_to_prune = int(len(candidates) * self.prune_amount)
        if self.prune_amount > 0 and num_to_prune == 0:
            num_to_prune = 1
        num_to_prune = min(num_to_prune, len(candidates))

        weakest = sorted(candidates, key=lambda x: x[0])[:num_to_prune]
        
        # Group pruned nodes by layer for efficient processing
        layer_to_nodes = {}
        for _, layer, node_idx in weakest:
            if layer not in layer_to_nodes:
                layer_to_nodes[layer] = []
            layer_to_nodes[layer].append(node_idx)
        
        # Prune nodes and their connections to the next layer
        for layer, node_indices in layer_to_nodes.items():
            # Zero out output weights and bias of pruned nodes
            for node_idx in node_indices:
                layer.weight_mask.data[node_idx, :] = 0
                if layer.bias is not None and hasattr(layer, "bias_mask"):
                    layer.bias_mask.data[node_idx] = 0
            
            # Find the next layer and zero out corresponding input connections
            layer_idx = hidden_layers.index(layer)
            if layer_idx + 1 < len(linear_layers):  # Check if there's a next layer
                next_layer = linear_layers[layer_idx + 1]
                
                # Initialize mask if not present
                if not hasattr(next_layer, "weight_mask"):
                    prune.custom_from_mask(next_layer, name="weight", mask=torch.ones_like(next_layer.weight))
                
                # Zero out input connections from pruned nodes
                for node_idx in node_indices:
                    next_layer.weight_mask.data[:, node_idx] = 0

        print(f"Pruned {num_to_prune} hidden nodes (with forward connections)")
