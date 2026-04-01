# Class representing full neural network
from .Dataset import Dataset
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import prune
import matplotlib.pyplot as plt

import copy

class NeuralNetwork:
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.num_memory_levels = num_memory_levels
        self.nn_model = self.get_model(model_type).to(self.device)
        self.forward_model = forward_model  # True if forward model, False if inverse model
        self.batch_size = batch_size
    
    def training_data(self, dataset):
        """Get aligned training data for NN model"""
        if self.forward_model:
            model_training_input, model_training_output = dataset.input_data, dataset.output_data
        else:
            model_training_input, model_training_output = dataset.output_data, dataset.input_data
        training_xfc = self.gen_input_feature(model_training_input)
        training_output_aligned = self.gen_output_feature(model_training_input, model_training_output) 

        return training_xfc, training_output_aligned
 
    
    def build_dataloaders(self, x, y, shuffle=False, num_workers=4, pin_memory=True):
        """Build dataloaders for dataset with parallel data loading"""
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        
        # Use multiple workers for parallel data loading and pin memory for faster GPU transfer
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
        return dataloader

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3, target_nmse = None):
        """Train model and return the best model based on validation loss"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        # Create dataloaders
        training_xfc, training_output_aligned = self.training_data(training_dataset)

        validation_xfc, validation_output_aligned = self.training_data(validation_dataset)

        train_loader = self.build_dataloaders(training_xfc, training_output_aligned)
        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned)
        for epoch in range(num_epochs):
            self.nn_model.train()
            running_train_loss = 0
            running_valid_loss = 0
            
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.nn_model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * xb.size(0)
                
            train_loss = running_train_loss
            
            self.nn_model.eval()
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.nn_model(xb)
                    loss = criterion(preds, yb)
                    running_valid_loss += loss.item() * xb.size(0)
                
            valid_loss = running_valid_loss
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(valid_loss)
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(self.nn_model.state_dict())
                best_epoch = epoch + 1
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if self.forward_model:
                    print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}  NMSE={self.calculate_forward_nmse(validation_dataset):.4f} dB")
                else:
                    print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}")

            # If the target nmse has been reached, break out of the training
            if target_nmse is not None and self.forward_model and self.calculate_forward_nmse(validation_dataset) < target_nmse:
                break
        
        # Load best model
        self.nn_model.load_state_dict(best_model_state)
        print(f"\nBest model from epoch {best_epoch} with validation loss: {best_valid_loss:.4e}")
        
        return train_losses, valid_losses, best_epoch
    
    def generate_model_output(self, x):
        """Generate phase denormalised output for given input x using trained NN model. Return both trimmed input and output"""
        self.nn_model.eval()
        with torch.no_grad():
            xfc = self.gen_input_feature(x)
            X = torch.tensor(xfc, dtype=torch.float32).to(self.device)
            preds = self.nn_model(X).detach().cpu().numpy()
        # Reconstruct complex output
        y_pred = preds[:, 0] + 1j * preds[:, 1]

        # Phase denormalise
        phase = Dataset.conj_phase(x)  #conj
        y_pred = y_pred * np.conj(phase[self.num_memory_levels:])
        return y_pred
    
    def calculate_forward_nmse(self, dataset):
        """Calculate NMSE for forward model on given dataset"""
        if not self.forward_model:
            raise ValueError("Model is not a forward model")
        y_true = dataset.output_data[self.num_memory_levels:]
        y_pred = self.generate_model_output(dataset.input_data)
        nmse = 10 * np.log10(np.sum(np.abs(y_true - y_pred)**2) / np.sum(np.abs(y_true)**2))
        return nmse
    
    
    def prune_model(self, parameters_to_prune_list, prune_amount=0.2):
        """Apply pruning to the model in-place"""
        parameters_to_prune = []
        named_modules = dict(self.nn_model.named_modules())

        for layer_name in parameters_to_prune_list:
            module = named_modules.get(layer_name)
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            available_linear_layers = [
                name for name, module in self.nn_model.named_modules()
                if isinstance(module, nn.Linear)
            ]
            raise ValueError(
                "No valid linear layers were selected for pruning. "
                f"Requested: {parameters_to_prune_list}. "
                f"Available linear layers: {available_linear_layers}"
            )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_amount,
        )

    def _calculate_initial_valid_loss(self, validation_dataset):
        # Calculate initial validation loss (for pruning experiments)
        validation_xfc, validation_output_aligned = self.training_data(validation_dataset)

        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned, shuffle=False)
        criterion = nn.MSELoss()
        self.nn_model.eval()
        with torch.no_grad():
            initial_valid_loss = 0
            for xb, yb in valid_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                preds = self.nn_model(xb)
                loss = criterion(preds, yb)
                initial_valid_loss += loss.item() * xb.size(0)
        return initial_valid_loss

    def _get_pruning_percentage(self):
        """Calculate the current percentage of pruned weights"""
        total_params = 0
        pruned_params = 0
        for name, module in self.nn_model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    pruned_params += (mask == 0).sum().item()
                else:
                    total_params += module.weight.numel()
        
        return (pruned_params / total_params * 100) if total_params > 0 else 0
    
    def get_num_params(self):
        total_params = 0
        for name, module in self.nn_model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                else:
                    total_params += module.weight.numel()
        
        return total_params
    
    @staticmethod
    def plot_valid_curve(valid_losses, best_epoch=None):
        """
        Plot validation loss curve
        """

        epochs = np.arange(1, len(valid_losses) + 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot validation curve
        ax.plot(
            epochs,
            valid_losses,
            linewidth=2,
            color='tab:blue',
            label='Validation Loss'
        )

        # Best epoch marker
        if best_epoch is not None:
            best_loss = valid_losses[best_epoch - 1]

            ax.axvline(
                x=best_epoch,
                linestyle='--',
                linewidth=1.5,
                color='tab:red',
                alpha=0.8,
                label=f'Best Epoch = {best_epoch}'
            )

            ax.plot(
                best_epoch,
                best_loss,
                marker='*',
                markersize=14,
                color='tab:red',
                markeredgecolor='black',
                markeredgewidth=1.2
            )

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(
            'Validation Loss vs Epoch',
            fontsize=14,
            fontweight='bold'
        )

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

                
class OneLayerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OneLayerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class OneLayerNetwork_Skip(nn.Module):
    """OneLayerNetwork with skip connection from input to first hidden layer output"""
    def __init__(self, input_size, hidden_size):
        super(OneLayerNetwork_Skip, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # Input is concatenated with hidden layer output, so input size becomes input_size + hidden_size
        self.fc2 = nn.Linear(input_size + hidden_size, 2)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        # Concatenate original input with hidden layer output (skip connection)
        h_skip = torch.cat([x, h], dim=1)
        return self.fc2(h_skip)

    
class ThreeLayerNetwork(nn.Module):    
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(ThreeLayerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class ThreeLayerNetwork_Skip(nn.Module):
    """Three-layer network with skip connections at each layer"""
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(ThreeLayerNetwork_Skip, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        # Skip connection: concat input (input_size) + hidden1 (hidden_size1)
        self.fc2 = nn.Linear(input_size + hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        # Skip connection: concat hidden1 (hidden_size1) + hidden2 (hidden_size2)
        self.fc3 = nn.Linear(hidden_size1 + hidden_size2, 2)

    def forward(self, x):
        x_orig = x
        h1 = self.relu1(self.fc1(x))
        # First skip: concatenate original input with first hidden layer
        h1_skip = torch.cat([x_orig, h1], dim=1)
        h2 = self.relu2(self.fc2(h1_skip))
        # Second skip: concatenate first hidden layer with second hidden layer
        h2_skip = torch.cat([h1, h2], dim=1)
        return self.fc3(h2_skip)


# Multi-layer network without skip connections
class MultiLayerNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MultiLayerNetwork, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiLayerNetwork_Skip(nn.Module):
    """Multi-layer network with skip connections from input to each hidden block and output."""
    def __init__(self, input_size, hidden_sizes):
        super(MultiLayerNetwork_Skip, self).__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        self.activation = nn.ReLU()

        in_size = input_size
        for idx, hidden_size in enumerate(hidden_sizes):
            layer_in = in_size if idx == 0 else in_size + input_size
            self.hidden_layers.append(nn.Linear(layer_in, hidden_size))
            in_size = hidden_size

        self.output_layer = nn.Linear(in_size + input_size, 2)

    def forward(self, x):
        x_orig = x
        h = x
        for idx, layer in enumerate(self.hidden_layers):
            if idx == 0:
                h = self.activation(layer(h))
            else:
                h = self.activation(layer(torch.cat([h, x_orig], dim=1)))
        h_out = torch.cat([h, x_orig], dim=1)
        return self.output_layer(h_out)