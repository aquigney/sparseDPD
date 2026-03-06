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
            model_training_output, model_training_input = dataset.input_data, dataset.output_data
        training_xfc = self.gen_input_feature(model_training_input)
        training_output_aligned = self.gen_output_feature(model_training_input, model_training_output) 

        return training_xfc, training_output_aligned
 
    
    def build_dataloaders(self, x, y, shuffle=False):
        """Build dataloaders for dataset"""
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3):
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

    def get_model(self, model_type='OneLayerNetwork'):
        """Return NN model instance"""
        input_size = self.num_memory_levels * 4 -2  # Real/Imaginary parts + A and A^3 features
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
    
    def gen_input_feature(self, x):
        """Generates features from input signal for NN model"""

        num_points = len(x)
        phase = Dataset.conj_phase(x) #conj
        I = np.real(x)
        Q = np.imag(x)

        phase_norm_data = np.zeros((num_points, self.num_memory_levels), dtype=complex)

        for n in range(self.num_memory_levels, num_points):
            for m in range(self.num_memory_levels): 
                phase_norm_data[n, m] = x[n - m] * phase[n]
                

        Ax = np.sqrt(I**2 + Q**2)
        A_feats = np.zeros((num_points, self.num_memory_levels))
        for n in range(self.num_memory_levels, num_points):
            for m in range(self.num_memory_levels):
                A_feats[n, m] = Ax[n - m]

        # Trim first num_memory_levels rows
        phase_norm_data = phase_norm_data[self.num_memory_levels:, :]
        A_feats = A_feats[self.num_memory_levels:, :]
        A3_feats = A_feats**3

        imag_pn = np.imag(phase_norm_data)[:, 1:]   # drop imag of current tap (m=0)
        A_taps  = A_feats[:, 1:]                   # drop A of current tap (m=0), keep tapped A only

        xfc = np.hstack([
            np.real(phase_norm_data),   # M
            imag_pn,                    # M-1
            A_taps,                     # M-1   <-- changed
            A3_feats,                   # M
        ]).astype(np.float32)

        return xfc
    
    def gen_output_feature(self, x, y):
        """Generates features from output signal for NN model"""
        y_norm = y * Dataset.conj_phase(x) 
        y_norm = y_norm[self.num_memory_levels:]
        return np.array([np.real(y_norm), np.imag(y_norm)]).T.astype(np.float32)
    
    

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
    
    

class PGJANET_NeuralNetwork(NeuralNetwork):
    def __init__(self, num_memory_levels, model_type='PGJANETNetwork', forward_model=False, batch_size=256, seq_len=None, seq_stride=None, hidden_size =16):
        super().__init__(num_memory_levels, model_type, forward_model, batch_size=batch_size)
        self.seq_len = seq_len if seq_len is not None else num_memory_levels
        self.seq_stride = seq_stride if seq_stride is not None else 1
        self.hidden_size = hidden_size

    def training_data(self, dataset):
        if self.forward_model:
            x_in, y_out = dataset.input_data, dataset.output_data
        else:
            y_out, x_in = dataset.input_data, dataset.output_data

        X, Y_last = self.make_windows_iq(x_in, y_out, self.seq_len)
        return X, Y_last
    
    def get_model(self, model_type='PGJANETNetwork'):
        """Return PGJANET model instance."""
        if model_type != 'PGJANETNetwork':
            print("Model type not recognized for PGJANET_NeuralNetwork")
            return None
        output_size = 2
        return PGJANETNetwork(hidden_size=self.hidden_size, output_size=output_size)

    def gen_input_feature(self, x, stride=None):
        return np.array([np.real(x), np.imag(x)]).T.astype(np.float32)
    
    def generate_model_output(self, x):
        """Generate output for given input x using trained NN model with windowing."""
        self.nn_model.eval()
        with torch.no_grad():
            # Create windows - only need input windows for prediction
            x_iq = np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)  # (N,2)
            N = x_iq.shape[0]
            T = self.seq_len
            
            if N < T:
                return np.array([], dtype=np.complex128)
            
            # Create input windows (N-T+1, T, 2)
            X_windows = np.stack([x_iq[i:i+T] for i in range(N - T + 1)], axis=0)
            X = torch.tensor(X_windows, dtype=torch.float32).to(self.device)
            preds = self.nn_model(X).detach().cpu().numpy()  # (N-T+1, T, 2)
            
            # Extract last timestep only (many-to-one prediction)
            preds = preds[:, -1, :]  # (N-T+1, 2)
        
        # Reconstruct complex output
        y_pred = preds[:, 0] + 1j * preds[:, 1]
        return y_pred
        
    
    def gen_output_feature(self, x, y):
        """Generates features from output signal for NN model"""
        return np.array([np.real(y), np.imag(y)]).T.astype(np.float32)

    
    def calculate_forward_nmse(self, dataset):
        """Calculate NMSE for forward model on given dataset.
        Account for windowing - output is trimmed by seq_len-1 samples."""
        if not self.forward_model:
            raise ValueError("Model is not a forward model")
        y_pred = self.generate_model_output(dataset.input_data)
        # Trim y_true to match windowed output (last seq_len-1 samples are used for windowing)
        y_true = dataset.output_data[self.seq_len - 1:]
        nmse = 10 * np.log10(np.sum(np.abs(y_true - y_pred)**2) / np.sum(np.abs(y_true)**2))
        return nmse
    
    def make_windows_iq(self, x_complex, y_complex, T):
        x_iq = np.stack([np.real(x_complex), np.imag(x_complex)], axis=-1).astype(np.float32)  # (N,2)
        y_iq = np.stack([np.real(y_complex), np.imag(y_complex)], axis=-1).astype(np.float32)  # (N,2)

        N = x_iq.shape[0]
        if N < T:
            return np.empty((0, T, 2), np.float32), np.empty((0, 2), np.float32)

        starts = range(0, N-T+1, self.seq_stride)
        X = np.stack([x_iq[i:i+T] for i in starts], axis=0)  # (N-T+1, T, 2)
        Y = np.stack([y_iq[i+T-1] for i in starts], axis=0)  # (N-T+1, T, 2)
        return X, Y
    
    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3):
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

        train_loader = self.build_dataloaders(training_xfc, training_output_aligned, shuffle=True)
        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned, shuffle=False)
        
        for epoch in range(num_epochs):
            self.nn_model.train()
            running_train_loss = 0
            running_valid_loss = 0
            
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds_seq = self.nn_model(xb)
                pre_last = preds_seq[:,-1,:]
                loss = criterion(pre_last, yb)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * xb.size(0)
                
            train_loss = running_train_loss
            
            self.nn_model.eval()
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds_seq = self.nn_model(xb)
                    pre_last = preds_seq[:,-1,:]
                    loss = criterion(pre_last, yb)
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
                print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}")
        
        # Load best model
        self.nn_model.load_state_dict(best_model_state)
        print(f"\nBest model from epoch {best_epoch} with validation loss: {best_valid_loss:.4e}")
        
        return train_losses, valid_losses, best_epoch


##### PyTorch models #####

class PGJANETNetwork(nn.Module):
    def __init__(self, hidden_size, output_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_a  = nn.Linear(hidden_size + 1, hidden_size, bias=bias)
        self.W_p1 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)
        self.W_p2 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)

        self.W_f = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)
        self.W_g = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)

        self.W_o = nn.Linear(hidden_size, output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x, h_0=None):
        # x: (B,T,2)
        if x.ndim != 3 or x.size(-1) != 2:
            raise ValueError(f"Expected x shaped (B,T,2). Got {tuple(x.shape)}")

        B, T, _ = x.shape

        if h_0 is None:
            # mimic (num_layers, B, H)
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            # accept (1,B,H) or (num_layers,B,H) -> use layer 0
            h = h_0[0]

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]  # (B,2)

            i_x = x_t[:, 0].unsqueeze(-1)
            q_x = x_t[:, 1].unsqueeze(-1)

            amp_x = torch.sqrt(torch.clamp(i_x**2 + q_x**2, min=1e-12))
            theta = torch.atan2(q_x, i_x)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            h_x   = torch.cat([h, amp_x], dim=-1)
            h_cos = torch.cat([h, cos_theta], dim=-1)
            h_sin = torch.cat([h, sin_theta], dim=-1)

            a_n  = torch.tanh(self.W_a(h_x))
            p1_n = torch.tanh(self.W_p1(h_cos))
            p2_n = torch.tanh(self.W_p2(h_sin))

            u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)

            h_u = torch.cat([h, u_n], dim=-1)

            f_n = torch.sigmoid(self.W_f(h_u))
            g_n = torch.tanh(self.W_g(h_u))

            h = f_n * h + (1 - f_n) * g_n

            y_t = self.W_o(h)         # (B,2)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (B,T,2)
        return outputs

    def reset_parameters(self):
        for module in [self.W_a, self.W_p1, self.W_p2, self.W_f, self.W_g, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


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