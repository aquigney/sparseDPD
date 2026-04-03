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

from .NeuralNetwork import NeuralNetwork

import copy

class PGJANET_NeuralNetwork(NeuralNetwork):
    def __init__(self, num_memory_levels, model_type='PGJANETNetwork', forward_model=False, batch_size=256, seq_len=None, seq_stride=None, hidden_size =16, nn_file_path=None):
        self.hidden_size = hidden_size  # Set this BEFORE super().__init__() because get_model() needs it
        super().__init__(num_memory_levels, model_type, forward_model, batch_size=batch_size)
        self.seq_len = seq_len if seq_len is not None else num_memory_levels
        self.seq_stride = seq_stride if seq_stride is not None else 1

        if nn_file_path is not None:
            self.load_nn_from_file(nn_file_path)

    def write_nn_to_file(self, file_path):
        """Persist PGJANET configuration and learned weights to a file."""
        # Make pruning permanent before saving to avoid loading issues
        self.make_pruning_permanent()
        
        payload = {
            "num_memory_levels": self.num_memory_levels,
            "model_type": "PGJANETNetwork",
            "forward_model": self.forward_model,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
            "hidden_size": self.hidden_size,
            "state_dict": self.nn_model.state_dict(),
        }
        torch.save(payload, file_path)

    def load_nn_from_file(self, file_path):
        """Load PGJANET configuration and weights from a saved file."""
        checkpoint = torch.load(file_path, map_location=self.device)

        if "state_dict" not in checkpoint:
            raise ValueError("Invalid PGJANET checkpoint: missing 'state_dict'.")

        saved_hidden_size = checkpoint.get("hidden_size", self.hidden_size)
        if saved_hidden_size != self.hidden_size:
            self.hidden_size = saved_hidden_size
            self.nn_model = self.get_model('PGJANETNetwork').to(self.device)

        #self.num_memory_levels = checkpoint.get("num_memory_levels", self.num_memory_levels)
        self.forward_model = checkpoint.get("forward_model", self.forward_model)
        #self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.seq_len = checkpoint.get("seq_len", self.seq_len)
        self.seq_stride = checkpoint.get("seq_stride", self.seq_stride)

        self.nn_model.load_state_dict(checkpoint["state_dict"])
        self.nn_model.eval()

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
        """Optimized window creation using vectorized operations."""
        x_iq = np.stack([np.real(x_complex), np.imag(x_complex)], axis=-1).astype(np.float32)  # (N,2)
        y_iq = np.stack([np.real(y_complex), np.imag(y_complex)], axis=-1).astype(np.float32)  # (N,2)

        N = x_iq.shape[0]
        if N < T:
            return np.empty((0, T, 2), np.float32), np.empty((0, 2), np.float32)

        # Use stride_tricks for efficient window creation (avoids copying data)
        from numpy.lib.stride_tricks import as_strided
        
        # Calculate number of windows
        num_windows = (N - T) // self.seq_stride + 1
        
        if self.seq_stride == 1:
            # Optimized path for stride=1 using as_strided (zero-copy)
            shape_x = (num_windows, T, 2)
            strides_x = (x_iq.strides[0], x_iq.strides[0], x_iq.strides[1])
            X = as_strided(x_iq, shape=shape_x, strides=strides_x).copy()  # Copy at end for safety
            
            # Y is just the last timestep of each window
            Y = y_iq[T-1:T-1+num_windows]
        else:
            # For non-unit stride, use optimized indexing
            starts = np.arange(0, N-T+1, self.seq_stride)
            idx = starts[:, None] + np.arange(T)  # Broadcasting: (num_windows, T)
            X = x_iq[idx]  # (num_windows, T, 2)
            Y = y_iq[starts + T - 1]  # (num_windows, 2)
        
        return X, Y
    
    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3, target_nmse=None):
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
                if self.forward_model:
                    print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}  NMSE={self.calculate_forward_nmse(validation_dataset):.4f} dB")
                else:
                    print(f"Epoch {epoch + 1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid Loss={valid_loss:.4e}  LR={current_lr:.2e}")

            if target_nmse is not None and self.forward_model and self.calculate_forward_nmse(validation_dataset) < target_nmse:
                break
        # Load best model
        self.nn_model.load_state_dict(best_model_state)
        print(f"\nBest model from epoch {best_epoch} with validation loss: {best_valid_loss:.4e}")
        
        return train_losses, valid_losses, best_epoch
    
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

                preds_seq = self.nn_model(xb)
                pre_last = preds_seq[:,-1,:]
                loss = criterion(pre_last, yb)
                initial_valid_loss += loss.item() * xb.size(0)
        return initial_valid_loss


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
