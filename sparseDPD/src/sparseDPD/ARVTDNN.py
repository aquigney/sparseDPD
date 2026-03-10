from .NeuralNetwork import NeuralNetwork, OneLayerNetwork, OneLayerNetwork_Skip, ThreeLayerNetwork, ThreeLayerNetwork_Skip, MultiLayerNetwork, MultiLayerNetwork_Skip
from .Dataset import Dataset
import numpy as np

import torch

class ARVTDNN_NeuralNetwork(NeuralNetwork):
    """Similar to PNTDNN but with no phase normalization"""
    def __init__(self, num_memory_levels, model_type='OneLayerNetwork', forward_model=False, nn_file_path=None):
        super().__init__(num_memory_levels, model_type, forward_model)
        self.model_type = model_type
        self.forward_model = forward_model
        
        if nn_file_path is not None:
            self.load_nn_from_file(nn_file_path)
        
    def write_nn_to_file(self, file_path):
        """Persist ARVTDNN configuration and learned weights to a file."""
        payload = {
            "num_memory_levels": self.num_memory_levels,
            "model_type": self.model_type,
            "forward_model": self.forward_model,
            "state_dict": self.nn_model.state_dict(),
        }
        torch.save(payload, file_path)

    def load_nn_from_file(self, file_path):
        """Load ARVTDNN configuration and weights from a saved file."""
        checkpoint = torch.load(file_path, map_location=self.device)

        if "state_dict" not in checkpoint:
            raise ValueError("Invalid ARVTDNN checkpoint: missing 'state_dict'.")

        self.model_type = checkpoint.get("model_type", self.model_type)
        self.num_memory_levels = checkpoint.get("num_memory_levels", self.num_memory_levels)
        self.forward_model = checkpoint.get("forward_model", self.forward_model)

        # Rebuild with restored config before loading weights.
        self.nn_model = self.get_model(self.model_type).to(self.device)

        self.nn_model.load_state_dict(checkpoint["state_dict"])
        self.nn_model.eval()

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

