from .NeuralNetwork import NeuralNetwork, OneLayerNetwork, OneLayerNetwork_Skip, ThreeLayerNetwork, ThreeLayerNetwork_Skip, MultiLayerNetwork, MultiLayerNetwork_Skip
from .Dataset import Dataset
import torch
import numpy as np



class PNTDNN_NeuralNetwork(NeuralNetwork):
    # This class is a wrapper around the base NeuralNetwork to specify the PNTDNN architecture and feature generation 
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False, nn_file_path=None):
        super(PNTDNN_NeuralNetwork, self).__init__(num_memory_levels=num_memory_levels, model_type=model_type, forward_model=forward_model)
        self.num_memory_levels = num_memory_levels
        self.model_type = model_type
        self.forward_model = forward_model
        
        if nn_file_path is not None:
            self.load_nn_from_file(nn_file_path)
        
    def write_nn_to_file(self, file_path):
        """Persist PNTDNN configuration and learned weights to a file."""
        payload = {
            "num_memory_levels": self.num_memory_levels,
            "model_type": self.model_type,
            "forward_model": self.forward_model,
            "state_dict": self.nn_model.state_dict(),
        }
        torch.save(payload, file_path)

    def load_nn_from_file(self, file_path):
        """Load PNTDNN configuration and weights from a saved file."""
        checkpoint = torch.load(file_path, map_location=self.device)

        if "state_dict" not in checkpoint:
            raise ValueError("Invalid PNTDNN checkpoint: missing 'state_dict'.")

        self.model_type = checkpoint.get("model_type", self.model_type)
        self.num_memory_levels = checkpoint.get("num_memory_levels", self.num_memory_levels)
        self.forward_model = checkpoint.get("forward_model", self.forward_model)

        # Rebuild with restored config before loading weights.
        self.nn_model = self.get_model(self.model_type).to(self.device)

        self.nn_model.load_state_dict(checkpoint["state_dict"])
        self.nn_model.eval()

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
        # For forward models: normalize by input phase (x)
        # For inverse models: normalize by output's own phase (y) so inference phase matches
        if self.forward_model:
            y_norm = y * Dataset.conj_phase(x)
        else:
            #y_norm = y * Dataset.conj_phase(y)
            y_norm = y * Dataset.conj_phase(x)
        y_norm = y_norm[self.num_memory_levels:]
        return np.array([np.real(y_norm), np.imag(y_norm)]).T.astype(np.float32)
    