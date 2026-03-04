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
        self.initial_state_dict = None  # For lottery ticket hypothesis

    def get_model(self, model_type='OneLayerNetwork'):
        """Return NN model instance"""
        input_size = self.num_memory_levels * 4  # Real/Imaginary parts + A and A^3 features
        if model_type == 'OneLayerNetwork':
            hidden_size = 12
            model = OneLayerNetwork(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'OneLayerNetwork_Skip':
            hidden_size = 12
            model = OneLayerNetwork_Skip(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'ThreeLayerNetwork':
            hidden_size1 = 30
            hidden_size2 = 15
            model = ThreeLayerNetwork(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2)
        elif model_type == 'ThreeLayerNetwork_Skip':
            hidden_size1 = 30
            hidden_size2 = 15
            model = ThreeLayerNetwork_Skip(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2)
        elif model_type == 'MultiLayerNetwork':
            hidden_sizes = [64, 32, 32, 32, 32, 16, 8]
            model = MultiLayerNetwork(input_size=input_size, hidden_sizes=hidden_sizes)
        elif model_type == 'MultiLayerNetwork_Skip':
            hidden_sizes = [64, 32, 32, 32, 32, 16, 8]
            model = MultiLayerNetwork_Skip(input_size=input_size, hidden_sizes=hidden_sizes)
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

    def _align_loss_tensors(self, preds, targets):
        """Align model predictions and targets before loss calculation.

        Default behavior is sequence-to-sequence / direct element-wise loss.
        Subclasses can override for model-specific training targets.
        """
        return preds, targets

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3, use_frames=False, frame_stride=1, frame_length=500, grad_clip_val=0.0):
        """Train model and return the best model based on validation loss.

        Args:
            grad_clip_val (float): max norm for gradient clipping; 0.0 disables clipping.
            reset_weights (bool): if True, reset weights to initial values before training
                                 (for lottery ticket hypothesis). Default is False.
        """
        # Reset weights if requested (for lottery ticket hypothesis)
        if reset_weights:
            self.reset_to_initial_weights()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6) #min_lr=1e-6?
        
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf') # original set as infinity to ensure any valid loss will be better
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
                loss_preds, loss_targets = self._align_loss_tensors(preds, yb)
                loss = criterion(loss_preds, loss_targets)
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
                    loss_preds, loss_targets = self._align_loss_tensors(preds, yb)
                    loss = criterion(loss_preds, loss_targets)
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
                loss_preds, loss_targets = self._align_loss_tensors(preds, yb)
                loss = criterion(loss_preds, loss_targets)
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
            return np.empty((0, 4 * M), dtype=np.float32)

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

        # Build features following previous layout
        real_pn = np.real(pn)               # (N-M+1, M)
        imag_pn = np.imag(pn)               # (N-M+1, M)
        A_taps = A                          # (N-M+1, M)

        xfc = np.hstack([
            real_pn,
            imag_pn,
            A_taps,
            A3
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
        feat_dim = 4 * self.num_memory_levels 

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

        real_pn = np.real(pn).reshape(total, self.num_memory_levels)
        imag_pn = np.imag(pn).reshape(total, self.num_memory_levels)
        A_taps = A.reshape(total, self.num_memory_levels)
        A3_r = A3.reshape(total, self.num_memory_levels)

        Xfc = np.hstack([real_pn, imag_pn, A_taps, A3_r]).astype(np.float32)

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
            return np.empty((0, 4 * M), dtype=np.float32)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x, window_shape=M)
        except Exception:
            windows = np.stack([x[i:N - M + i + 1] for i in range(M)], axis=1)

        windows = windows[1:]

        taps = windows[:, ::-1]

        A = np.abs(taps)
        A3 = A ** 3

        real_taps = np.real(taps)               # (N-M+1, M)
        imag_taps = np.imag(taps)               # (N-M+1, M)
        A_taps = A                              # (N-M+1, M)

        xfc = np.hstack([
            real_taps,
            imag_taps,
            A_taps,
            A3
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
        feat_dim = 4 * self.num_memory_levels

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

        real_taps = np.real(taps).reshape(total, self.num_memory_levels)
        imag_taps = np.imag(taps).reshape(total, self.num_memory_levels)
        A_taps = A.reshape(total, self.num_memory_levels)
        A3_r = A3.reshape(total, self.num_memory_levels)

        Xfc = np.hstack([real_taps, imag_taps, A_taps, A3_r]).astype(np.float32)

        # Slice output to match the skipped first window (same as gen_output_feature: y[self.num_memory_levels:])
        y_curr = y_frames[:, self.num_memory_levels:].reshape(total)

        Yout = np.empty((total, 2), dtype=np.float32)
        Yout[:, 0] = np.real(y_curr).astype(np.float32)
        Yout[:, 1] = np.imag(y_curr).astype(np.float32)

        return torch.from_numpy(Xfc), torch.from_numpy(Yout)
    

class PGJANET_NeuralNetwork(NeuralNetwork):
    """PGJANET recurrent network wrapper using sequence IQ features."""
    def __init__(self, num_memory_levels, model_type='PGJANETNetwork', forward_model=False):
        # For PGJANET, num_memory_levels acts as the sequence length
        super().__init__(num_memory_levels, model_type, forward_model)

    def get_model(self, model_type='PGJANETNetwork'):
        """Return PGJANET model instance."""
        if model_type != 'PGJANETNetwork':
            print("Model type not recognized for PGJANET_NeuralNetwork")
            return None
        hidden_size = 15
        output_size = 2
        return PGJANETNetwork(hidden_size=hidden_size, output_size=output_size)

    def gen_input_feature(self, x):
        """Generate sequence IQ features shaped (N_valid, seq_len, 2).
        
        Args:
            x: Complex input array of shape (N,)
            
        Returns:
            Array of shape (N_valid, M, 2) where M is num_memory_levels (sequence length)
        """
        x = np.asarray(x)
        N = x.shape[0]
        M = int(self.num_memory_levels)

        if N < M:
            return np.empty((0, M, 2), dtype=np.float32)

        # Create sliding windows of IQ samples
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(x, window_shape=M)
        except Exception:
            windows = np.stack([x[i:N - M + i + 1] for i in range(M)], axis=1)

        # Skip first window to align with output (like other implementations)
        windows = windows[1:]  # Shape: (N-M, M)

        # Convert to IQ format: (N_valid, M, 2)
        x_seq = np.empty((windows.shape[0], M, 2), dtype=np.float32)
        x_seq[:, :, 0] = np.real(windows).astype(np.float32)  # I component
        x_seq[:, :, 1] = np.imag(windows).astype(np.float32)  # Q component
        
        return x_seq

    def gen_output_feature(self, x, y):
        """Generate aligned sequence output targets shaped (N_valid, seq_len, 2).
        
        For sequence-to-sequence training, returns all timesteps.
        
        Args:
            x: Complex input array (for alignment)
            y: Complex output array
            
        Returns:
            Array of shape (N_valid, M, 2) for sequence-to-sequence training
        """
        y = np.asarray(y)
        N = y.shape[0]
        M = int(self.num_memory_levels)

        if N < M:
            return np.empty((0, M, 2), dtype=np.float32)

        # Create sliding windows for output
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            y_windows = sliding_window_view(y, window_shape=M)
        except Exception:
            y_windows = np.stack([y[i:N - M + i + 1] for i in range(M)], axis=1)

        # Skip first window to align with input
        y_windows = y_windows[1:]  # Shape: (N-M, M)

        # Convert to IQ format for all timesteps: (N_valid, M, 2)
        y_seq = np.empty((y_windows.shape[0], M, 2), dtype=np.float32)
        y_seq[:, :, 0] = np.real(y_windows).astype(np.float32)
        y_seq[:, :, 1] = np.imag(y_windows).astype(np.float32)
        
        return y_seq

    def generate_model_output(self, x):
        """Generate unnormalized output using sequence-to-sequence predictions.
        
        For inference, we extract the last timestep from each sequence prediction.
        """
        self.nn_model.eval()
        with torch.no_grad():
            xseq = self.gen_input_feature(x)  # (N_valid, M, 2)
            X = torch.tensor(xseq, dtype=torch.float32).to(self.device)
            preds = self.nn_model(X).detach().cpu().numpy()  # (N_valid, M, 2)
            # Use last timestep for inference alignment
            preds_last = preds[:, -1, :]  # (N_valid, 2)

        return preds_last[:, 0] + 1j * preds_last[:, 1]

    def _align_loss_tensors(self, preds, targets):
        """Many-to-one training for PGJANET: compute loss on last timestep only."""
        if preds.ndim == 3 and targets.ndim == 3:
            return preds[:, -1, :], targets[:, -1, :]
        return preds, targets


##### PyTorch models #####

class PGJANETNetwork(nn.Module):
    """PGJANET recurrent cell for PA modeling with amplitude and phase gating.
    
    Based on OpenDPD's PGJANET architecture with gates conditioned on
    amplitude and phase (cos/sin) components of the input signal.
    """
    def __init__(self, hidden_size, output_size, bias=True):
        super(PGJANETNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        # Amplitude and phase gates (input concatenated with hidden state)
        self.W_a = nn.Linear(hidden_size + 1, hidden_size, bias=bias)   # Amplitude gate
        self.W_p1 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)  # Cosine phase gate
        self.W_p2 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)  # Sine phase gate
        
        # Processing gates
        self.W_f = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)  # Forget gate
        self.W_g = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)  # Candidate gate
        
        # Output projection
        self.W_o = nn.Linear(hidden_size, output_size, bias=bias)
        
        self.reset_parameters()

    def forward(self, x, h_0=None):
        """Forward pass through PGJANET cell.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 2) where last dim is [I, Q]
            h_0: Initial hidden state (optional)
            
        Returns:
            Tensor of shape (batch_size, seq_len, output_size) with predictions for all timesteps
        """
        if x.ndim != 3 or x.size(-1) != 2:
            raise ValueError(f"PGJANETNetwork expects x shaped (B,T,2). Got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            if h_0.ndim == 3:
                h = h_0[0]  # Take first layer if multi-layer format
            elif h_0.ndim == 2:
                h = h_0
            else:
                raise ValueError(f"h_0 must have shape (B,H) or (L,B,H). Got {tuple(h_0.shape)}")

        outputs = []
        
        # Process sequence timestep by timestep
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, 2)
            
            # Extract I and Q components
            i_x = x_t[:, 0].unsqueeze(-1)  # (batch_size, 1)
            q_x = x_t[:, 1].unsqueeze(-1)  # (batch_size, 1)
            
            # Calculate amplitude
            amp_x = torch.sqrt(torch.clamp(i_x**2 + q_x**2, min=1e-12))
            
            # Calculate phase components
            theta = torch.atan2(q_x, i_x)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # Concatenate hidden state with amplitude/phase features for gates
            h_x = torch.cat([h, amp_x], dim=-1)        # (batch_size, hidden_size + 1)
            h_cos = torch.cat([h, cos_theta], dim=-1)  # (batch_size, hidden_size + 1)
            h_sin = torch.cat([h, sin_theta], dim=-1)  # (batch_size, hidden_size + 1)

            # Compute amplitude and phase gates
            a_n = torch.tanh(self.W_a(h_x))
            p1_n = torch.tanh(self.W_p1(h_cos))
            p2_n = torch.tanh(self.W_p2(h_sin))

            # Compute modulation signal u_n (element-wise product with complements)
            u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)

            # Concatenate hidden state with modulation signal
            h_u = torch.cat([h, u_n], dim=-1)  # (batch_size, 2*hidden_size)

            # Compute forget and candidate gates
            f_n = torch.sigmoid(self.W_f(h_u))
            g_n = torch.tanh(self.W_g(h_u))

            # Update hidden state (GRU-like update)
            h = f_n * h + (1 - f_n) * g_n
            
            # Compute output for this timestep
            y_n = self.W_o(h)
            outputs.append(y_n)

        # Stack outputs to get (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def reset_parameters(self):
        for module in [self.W_a, self.W_p1, self.W_p2, self.W_f, self.W_g, self.W_o]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


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