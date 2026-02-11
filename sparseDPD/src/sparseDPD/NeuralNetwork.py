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
    def __init__(self, num_memory_levels, model_type='PNTDNN', forward_model=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.num_memory_levels = num_memory_levels
        self.nn_model = self.get_model(model_type).to(self.device)
        self.forward_model = forward_model  # True if forward model, False if inverse model

    def get_model(self, model_type='PNTDNN'):
        """Return NN model instance"""
        input_size = self.num_memory_levels * 5 - 2 # Real and Imaginary parts + A and A^3 features
        if model_type == 'PNTDNN':
            hidden_size = 15
            model = PNTDNN(input_size=input_size, hidden_size=hidden_size)

        elif model_type == 'PNTDNN_3_layers':
            hidden_size1 = 30
            hidden_size2 = 15
            model = PNTDNN_3_layers(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2)
        elif model_type == 'PNTDNN_Deep':
            hidden_sizes = [64, 32, 32, 32, 32, 16, 8]
            model = PNTDNN_Deep(input_size=input_size, hidden_sizes=hidden_sizes)
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
        A5_feats = A_feats**5

        imag_pn = np.imag(phase_norm_data)[:, 1:]   # drop imag of current tap (m=0)
        A_taps  = A_feats[:, 1:]                   # drop A of current tap (m=0), keep tapped A only

        xfc = np.hstack([
            np.real(phase_norm_data),   # M
            imag_pn,                    # M-1
            A_taps,                     # M-1   <-- changed
            A3_feats,                   # M
            A5_feats                    # M
        ]).astype(np.float32)

        return xfc
    
    def gen_output_feature(self, x, y):
        """Generates features from output signal for NN model"""
        y_norm = y * Dataset.conj_phase(x) # Normalised Output data TODO check if this breaks
        y_norm = y_norm[self.num_memory_levels:]
        return np.array([np.real(y_norm), np.imag(y_norm)]).T.astype(np.float32)
    
    def training_data(self, dataset):
        """Get aligned training data for NN model"""
        if self.forward_model:
            model_training_input, model_training_output = dataset.input_data, dataset.output_data
        else:
            model_training_output, model_training_input = dataset.input_data, dataset.output_data
        training_xfc = self.gen_input_feature(model_training_input)
        training_output_aligned = self.gen_output_feature(model_training_input, model_training_output) 

        return training_xfc, training_output_aligned
 
    
    def build_dataloaders(self, x, y, batch_size=256):
        """Build dataloaders for dataset"""
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3, use_frames=False, frame_stride=1, frame_length=500):
        """Train model and return the best model based on validation loss"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)
        
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        validation_xfc, validation_output_aligned = self.training_data(validation_dataset)
        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned)

        # Create dataloaders
        if not use_frames:
            training_xfc, training_output_aligned = self.training_data(training_dataset)
            train_loader = self.build_dataloaders(training_xfc, training_output_aligned)
            
        else: 
            train_ds = IQFrameTDNNSampleDataset(
                training_dataset.input_data,
                training_dataset.output_data,
                frame_length=frame_length,
                stride=frame_stride,
                mem_depth=self.num_memory_levels
            )

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

        valid_loader = self.build_dataloaders(validation_xfc, validation_output_aligned)
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
    
class PNTDNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PNTDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    


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
    

class IQFrameTDNNSampleDataset(torch.utils.data.Dataset):
    """
    Precomputed training dataset:
    - frames x and y into aligned [Nf, T]
    - generates samples for every valid t within each frame: t = (M-1) .. (T-1)
    - MATERIALIZES xfc and y_out ONCE in __init__
    - __getitem__ becomes cheap tensor indexing
    """

    def __init__(self, x, y, frame_length=500, stride=1, mem_depth=10):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.T = int(frame_length)
        self.S = int(stride)
        self.M = int(mem_depth)

        if self.T < self.M:
            raise ValueError(f"frame_length (T={self.T}) must be >= mem_depth (M={self.M})")

        # Build aligned frames (same indexing for x and y)
        x_frames = self._get_frames(self.x, self.T, self.S)  # [Nf, T]
        y_frames = self._get_frames(self.y, self.T, self.S)  # [Nf, T]
        n_frames = x_frames.shape[0]

        valid_t0 = self.M - 1
        n_valid = self.T - valid_t0
        total = n_frames * n_valid

        feat_dim = 5 * self.M - 2

        # Preallocate arrays (this is the key speedup)
        Xfc = np.empty((total, feat_dim), dtype=np.float32)
        Yout = np.empty((total, 2), dtype=np.float32)

        row = 0
        for k in range(n_frames):
            xf = x_frames[k]
            yf = y_frames[k]

            for t in range(valid_t0, self.T):
                taps = xf[t - (self.M - 1): t + 1][::-1]  # [M], order current..past

                x_curr = taps[0]
                c = np.exp(-1j * np.angle(x_curr))        # scalar complex

                pn = taps * c
                A = np.abs(taps)
                A3 = A ** 3
                A5 = A ** 5

                Xfc[row, :] = np.hstack([
                    np.real(pn),          # M
                    np.imag(pn)[1:],      # M-1
                    A[1:],                # M-1
                    A3,                   # M
                    A5                    # M
                ]).astype(np.float32)

                y_norm = yf[t] * c
                Yout[row, 0] = np.float32(np.real(y_norm))
                Yout[row, 1] = np.float32(np.imag(y_norm))

                row += 1

        # Convert ONCE to torch tensors (CPU)
        self.X = torch.from_numpy(Xfc)
        self.Y = torch.from_numpy(Yout)

    @staticmethod
    def _get_frames(sequence, frame_length, stride):
        n = len(sequence)
        n_frames = (n - frame_length) // stride + 1
        return np.stack([sequence[i*stride:i*stride+frame_length] for i in range(n_frames)])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
