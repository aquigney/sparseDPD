# Class representing full neural network
from .Dataset import Dataset
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import prune

import copy

class NeuralNetwork:
    def __init__(self, num_memory_levels, model_type='PNTDNN', forward_model=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")
        self.num_memory_levels = num_memory_levels
        self.nn_model = self.get_model(model_type)
        self.forward_model = forward_model  # True if forward model, False if inverse model
        

    def get_model(self, model_type='PNTDNN'):
        """Return NN model instance"""
        input_size = self.num_memory_levels * 4  # Real and Imaginary parts + A and A^3 features
        if model_type == 'PNTDNN':
            hidden_size = 15
            model = PNTDNN(input_size=input_size, hidden_size=hidden_size)
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

        xfc = np.hstack([np.real(phase_norm_data), np.imag(phase_norm_data), A_feats, A3_feats]).astype(np.float32)
        return xfc
    
    def gen_output_feature(self, y):
        """Generates features from output signal for NN model"""
        y_norm = y * Dataset.conj_phase(y) # Normalised Output data TODO check if this breaks
        y_norm = y_norm[self.num_memory_levels:]
        return np.array([np.real(y_norm), np.imag(y_norm)]).T.astype(np.float32)
    
    def training_data(self, dataset):
        """Get aligned training data for NN model"""
        if self.forward_model:
            model_training_input, model_training_output = dataset.input_data, dataset.output_data
        else:
            model_training_output, model_training_input = dataset.input_data, dataset.output_data
        training_xfc = self.gen_input_feature(model_training_input)
        training_output_aligned = self.gen_output_feature(model_training_output) 

        return training_xfc, training_output_aligned
 
    
    def build_dataloaders(self, x, y, batch_size=256):
        """Build dataloaders for dataset"""
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
        """Generate output for given input x using trained NN model. Return both trimmed input and output"""
        self.nn_model.eval()
        with torch.no_grad():
            xfc = self.gen_input_feature(x)
            X = torch.tensor(xfc, dtype=torch.float32)
            preds = self.nn_model(X).numpy()
        # Reconstruct complex output
        y_pred = preds[:, 0] + 1j * preds[:, 1]
        return y_pred
    
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