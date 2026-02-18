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
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.num_memory_levels = num_memory_levels
        self.nn_model = self.get_model(model_type).to(self.device)
        self.forward_model = forward_model  # True if forward model, False if inverse model

    def get_model(self, model_type='OneLayerNetwork'):
        """Return NN model instance"""
        input_size = self.num_memory_levels * 5 - 2 # Real and Imaginary parts + A and A^3 features
        if model_type == 'OneLayerNetwork':
            hidden_size = 12
            model = OneLayerNetwork(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'OneLayerNetwork_Skip':
            hidden_size = 12
            model = OneLayerNetwork_Skip(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'PNTDNN_3_layers':
            hidden_size1 = 30
            hidden_size2 = 15
            model = PNTDNN_3_layers(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2)
        elif model_type == 'PNTDNN_3_layers_Skip':
            hidden_size1 = 30
            hidden_size2 = 15
            model = PNTDNN_3_layers_Skip(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2)
        elif model_type == 'PNTDNN_Deep':
            hidden_sizes = [64, 32, 32, 32, 32, 16, 8]
            model = PNTDNN_Deep(input_size=input_size, hidden_sizes=hidden_sizes)
        else:
            print("Model type not recognized")
            model = None
        return model
    
    def training_data(self, dataset):
        """Get aligned training data for NN model"""
        if self.forward_model:
            model_training_input, model_training_output = dataset.input_data, dataset.output_data
        else:
            model_training_output, model_training_input = dataset.input_data, dataset.output_data
        training_xfc = self.gen_input_feature(model_training_input)
        training_output_aligned = self.gen_output_feature(model_training_input, model_training_output) 

        return training_xfc, training_output_aligned
 
    
    def build_dataloaders(self, x, y, batch_size=256, shuffle=False):
        """Build dataloaders for dataset"""
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3, use_frames=False, frame_stride=1, frame_length=500, grad_clip_val=0.0):
        """Train model and return the best model based on validation loss.

        Args:
            grad_clip_val (float): max norm for gradient clipping; 0.0 disables clipping.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6) #min_lr=1e-6?
        
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        validation_xfc, validation_output_aligned = self.training_data(validation_dataset)
        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned, shuffle=False)

        # Create dataloaders
        if not use_frames:
            training_xfc, training_output_aligned = self.training_data(training_dataset)
            train_loader = self.build_dataloaders(training_xfc, training_output_aligned, shuffle=True)
            
        else: 
            x_frames = self._get_frames(training_dataset.input_data, frame_length, frame_stride)
            y_frames = self._get_frames(training_dataset.output_data, frame_length, frame_stride)
            X, Y = self._precompute_iq_features(x_frames, y_frames)
            train_ds = TensorDataset(X, Y)
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
        
        self.nn_model.to(self.device)
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
                # Optional gradient clipping
                if grad_clip_val and grad_clip_val > 0.0:
                    nn.utils.clip_grad_norm_(self.nn_model.parameters(), grad_clip_val)
                optimizer.step()
                running_train_loss += loss.item() * xb.size(0)

            # Average epoch training loss
            try:
                n_train_samples = len(train_loader.dataset)
                train_loss = running_train_loss / float(n_train_samples)
            except Exception:
                train_loss = running_train_loss
            
            self.nn_model.eval()
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.nn_model(xb)
                    loss = criterion(preds, yb)
                    running_valid_loss += loss.item() * xb.size(0)

            # Average validation loss
            try:
                n_valid_samples = len(valid_loader.dataset)
                valid_loss = running_valid_loss / float(n_valid_samples)
            except Exception:
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
                print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}")
        
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

        if "fc1" in parameters_to_prune_list:
            parameters_to_prune.append((self.nn_model.fc1, 'weight'))
        if "fc2" in parameters_to_prune_list:
            parameters_to_prune.append((self.nn_model.fc2, 'weight'))
        if "fc3" in parameters_to_prune_list:
            parameters_to_prune.append((self.nn_model.fc3, 'weight'))
        
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


class PNTDNN_NeuralNetwork(NeuralNetwork):
    # This class is a wrapper around the base NeuralNetwork to specify the PNTDNN architecture and feature generation 
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False):
        super(PNTDNN_NeuralNetwork, self).__init__(num_memory_levels=num_memory_levels, model_type=model_type, forward_model=forward_model)
        self.num_memory_levels = num_memory_levels
        self.model_type = model_type
        self.forward_model = forward_model

    def gen_input_feature(self, x):
        """Generates features from input signal for NN model"""
        x = np.asarray(x)
        N = x.shape[0]
        M = int(self.num_memory_levels)
        if N < M:
            # no valid samples
            return np.empty((0, 5 * M - 2), dtype=np.float32)

        # sliding windows of length M: windows[i] = [x[i], ..., x[i+M-1]]
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x, window_shape=M)
        except Exception:
            # Fallback: construct with a simple stack (less efficient)
            windows = np.stack([x[i:N - M + i + 1] for i in range(M)], axis=1)

        # Original behavior aligned features with outputs starting at index self.num_memory_levels
        # windows shape is (N-M+1, M) corresponding to end indices (M-1 .. N-1).
        # To match previous code (which started at n = M .. N-1) we skip the first window.
        windows = windows[1:]

        # Each row corresponds to time index n = M .. (N-1); taps ordered current..past
        taps = windows[:, ::-1]

        # phase at each output time = conj_phase(x)[M:]
        phase = Dataset.conj_phase(x)[M:]

        # apply phase normalization per-row
        pn = taps * phase[:, None]

        A = np.abs(taps)
        A3 = A ** 3
        A5 = A ** 5

        # Build features following previous layout
        real_pn = np.real(pn)               # (N-M+1, M)
        imag_pn = np.imag(pn)[:, 1:]        # (N-M+1, M-1)
        A_taps = A[:, 1:]                   # (N-M+1, M-1)

        xfc = np.hstack([
            real_pn,
            imag_pn,
            A_taps,
            A3,
            A5
        ]).astype(np.float32)

        return xfc
    
    def gen_output_feature(self, x, y):
        """Generates features from output signal for NN model"""
        y_norm = y * Dataset.conj_phase(x) 
        y_norm = y_norm[self.num_memory_levels:]
        return np.array([np.real(y_norm), np.imag(y_norm)]).T.astype(np.float32)
    
    @staticmethod
    def _get_frames(sequence, frame_length, stride):
        """Extract frames from sequence"""
        n = len(sequence)
        n_frames = (n - frame_length) // stride + 1
        return np.stack([sequence[i*stride:i*stride+frame_length] for i in range(n_frames)])
    
    def _precompute_iq_features(self, x_frames, y_frames):
        """Precompute IQ frame features and targets"""
        n_frames = x_frames.shape[0]
        valid_t0 = self.num_memory_levels - 1
        n_valid = x_frames.shape[1] - valid_t0
        total = n_frames * n_valid
        feat_dim = 5 * self.num_memory_levels - 2

        if total == 0:
            return torch.empty((0, feat_dim), dtype=torch.float32), torch.empty((0, 2), dtype=torch.float32)

        # Vectorized extraction: build sliding windows
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x_frames, window_shape=self.num_memory_levels, axis=1)
        except Exception:
            windows = np.stack([x_frames[:, i:i + n_valid] for i in range(self.num_memory_levels)], axis=2)

        taps = windows[:, :, ::-1]
        c = np.exp(-1j * np.angle(taps[:, :, 0]))

        pn = taps * c[:, :, None]
        A = np.abs(taps)
        A3 = A ** 3
        A5 = A ** 5

        real_pn = np.real(pn).reshape(total, self.num_memory_levels)
        imag_pn = np.imag(pn)[:, :, 1:].reshape(total, self.num_memory_levels - 1)
        A_taps = A[:, :, 1:].reshape(total, self.num_memory_levels - 1)
        A3_r = A3.reshape(total, self.num_memory_levels)
        A5_r = A5.reshape(total, self.num_memory_levels)

        Xfc = np.hstack([real_pn, imag_pn, A_taps, A3_r, A5_r]).astype(np.float32)

        y_curr = y_frames[:, valid_t0:].reshape(total)
        c_flat = c.reshape(total)
        y_norm = y_curr * c_flat

        Yout = np.empty((total, 2), dtype=np.float32)
        Yout[:, 0] = np.real(y_norm).astype(np.float32)
        Yout[:, 1] = np.imag(y_norm).astype(np.float32)

        return torch.from_numpy(Xfc), torch.from_numpy(Yout)
    

class ARVTDNN_NeuralNetwork(NeuralNetwork):
    """Similar to PNTDNN but with no phase normalization"""
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False):
        super().__init__(num_memory_levels, model_type, forward_model)

    def gen_input_feature(self, x):
        """Generates features from input signal for NN model"""
        x = np.asarray(x)
        N = x.shape[0]
        M = int(self.num_memory_levels)
        if N < M:
            return np.empty((0, 5 * M - 2), dtype=np.float32)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x, window_shape=M)
        except Exception:
            windows = np.stack([x[i:N - M + i + 1] for i in range(M)], axis=1)

        windows = windows[1:]

        taps = windows[:, ::-1]

        A = np.abs(taps)
        A3 = A ** 3
        A5 = A ** 5

        real_taps = np.real(taps)               # (N-M+1, M)
        imag_taps = np.imag(taps)[:, 1:]        # (N-M+1, M-1)
        A_taps = A[:, 1:]                   # (N-M+1, M-1)

        xfc = np.hstack([
            real_taps,
            imag_taps,
            A_taps,
            A3,
            A5
        ]).astype(np.float32)

        return xfc
    
    def gen_output_feature(self, x, y):
        """Generates features from output signal for NN model"""
        y_curr = y[self.num_memory_levels:]
        return np.array([np.real(y_curr), np.imag(y_curr)]).T.astype(np.float32)
    
    def generate_model_output(self, x):
        """Generate unnormalized output for given input x using trained NN model. 
        Unlike PNTDNN, this returns raw (non-phase-normalized) predictions."""
        self.nn_model.eval()
        with torch.no_grad():
            xfc = self.gen_input_feature(x)
            X = torch.tensor(xfc, dtype=torch.float32).to(self.device)
            preds = self.nn_model(X).detach().cpu().numpy()
        # Reconstruct complex output (no phase denormalization needed)
        y_pred = preds[:, 0] + 1j * preds[:, 1]
        return y_pred
    
    @staticmethod
    def _get_frames(sequence, frame_length, stride):
        """Extract frames from sequence"""
        n = len(sequence)
        n_frames = (n - frame_length) // stride + 1
        return np.stack([sequence[i*stride:i*stride+frame_length] for i in range(n_frames)])
    
    def _precompute_iq_features(self, x_frames, y_frames):
        """Precompute IQ frame features and targets (without phase normalization)"""
        n_frames = x_frames.shape[0]
        n_valid = x_frames.shape[1] - self.num_memory_levels
        total = n_frames * n_valid
        feat_dim = 5 * self.num_memory_levels - 2

        if total == 0:
            return torch.empty((0, feat_dim), dtype=torch.float32), torch.empty((0, 2), dtype=torch.float32)

        # Vectorized extraction: build sliding windows
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x_frames, window_shape=self.num_memory_levels, axis=1)
        except Exception:
            windows = np.stack([x_frames[:, i:i + n_valid + 1] for i in range(self.num_memory_levels)], axis=2)

        # Skip the first window, same as gen_input_feature does (windows = windows[1:])
        windows = windows[:, 1:, :]
        
        taps = windows[:, :, ::-1]
        A = np.abs(taps)
        A3 = A ** 3
        A5 = A ** 5

        real_taps = np.real(taps).reshape(total, self.num_memory_levels)
        imag_taps = np.imag(taps)[:, :, 1:].reshape(total, self.num_memory_levels - 1)
        A_taps = A[:, :, 1:].reshape(total, self.num_memory_levels - 1)
        A3_r = A3.reshape(total, self.num_memory_levels)
        A5_r = A5.reshape(total, self.num_memory_levels)

        Xfc = np.hstack([real_taps, imag_taps, A_taps, A3_r, A5_r]).astype(np.float32)

        # Slice output to match the skipped first window (same as gen_output_feature: y[self.num_memory_levels:])
        y_curr = y_frames[:, self.num_memory_levels:].reshape(total)

        Yout = np.empty((total, 2), dtype=np.float32)
        Yout[:, 0] = np.real(y_curr).astype(np.float32)
        Yout[:, 1] = np.imag(y_curr).astype(np.float32)

        return torch.from_numpy(Xfc), torch.from_numpy(Yout)



##### PyTorch models #####

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

    
class PNTDNN_3_layers(nn.Module):    
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(PNTDNN_3_layers, self).__init__()
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


class PNTDNN_3_layers_Skip(nn.Module):
    """PNTDNN_3_layers with skip connections at each layer"""
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(PNTDNN_3_layers_Skip, self).__init__()
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


# Forward Neural Network model. It will take the same inputs, but contain many layers and neurons
class PNTDNN_Deep(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(PNTDNN_Deep, self).__init__()
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