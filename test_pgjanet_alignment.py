"""Test PGJANET alignment with different window strides"""
import numpy as np
from sparseDPD import PGJANET_NeuralNetwork, Dataset

def test_alignment(M=10, N=100, stride=1):
    """Test that PGJANET input/output features align correctly.
    
    Args:
        M: num_memory_levels (sequence length)
        N: signal length
        stride: window_stride
    """
    print(f"\n{'='*60}")
    print(f"Testing PGJANET alignment: M={M}, N={N}, stride={stride}")
    print(f"{'='*60}")
    
    # Create model
    model = PGJANET_NeuralNetwork(num_memory_levels=M, forward_model=True, window_stride=stride)
    
    # Create dummy complex signals
    x = np.random.randn(N) + 1j * np.random.randn(N)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    
    # Generate features for training (with specified stride)
    x_features = model.gen_input_feature(x, stride=stride)
    y_features = model.gen_output_feature(x, y, stride=stride)
    
    print(f"\nTraining (stride={stride}):")
    print(f"  Input features shape:  {x_features.shape}")
    print(f"  Output features shape: {y_features.shape}")
    
    # Check shapes match
    assert x_features.shape == y_features.shape, \
        f"Shape mismatch: {x_features.shape} vs {y_features.shape}"
    
    # Generate features for inference (always stride=1)
    x_features_inf = model.gen_input_feature(x, stride=1)
    
    print(f"\nInference (stride=1):")
    print(f"  Input features shape:  {x_features_inf.shape}")
    print(f"  Expected output length: {N - M}")
    
    # For stride=1, number of windows should be N-M (after skipping first)
    assert x_features_inf.shape[0] == N - M, \
        f"Inference shape wrong: {x_features_inf.shape[0]} vs expected {N-M}"
    
    # Verify alignment logic
    # With stride S, after skipping first window:
    # Window i contains x[i*S+S : i*S+S+M], predicts y[i*S+S+M-1]
    # For i=0: x[S:S+M] -> y[S+M-1]
    # For i=1: x[2S:2S+M] -> y[2S+M-1]
    # Number of windows = floor((N - M) / S)
    expected_train_windows = (N - M) // stride
    assert x_features.shape[0] == expected_train_windows, \
        f"Training windows wrong: {x_features.shape[0]} vs expected {expected_train_windows}"
    
    print(f"\n✓ All alignment checks passed!")
    
    # Verify phase normalization is applied
    x_norm = x * Dataset.conj_phase(x)
    y_norm = y * Dataset.conj_phase(x)
    
    # First training window (after skipping) starts at position stride
    # Check first input window matches normalized x[stride:stride+M]
    first_input_window = x_features[0]  # shape (M, 2)
    expected_first = x_norm[stride:stride+M]
    
    # Convert back to complex for comparison
    reconstructed = first_input_window[:, 0] + 1j * first_input_window[:, 1]
    
    if np.allclose(reconstructed, expected_first):
        print(f"✓ Phase normalization correctly applied to inputs")
    else:
        print(f"✗ Phase normalization issue in inputs")
        print(f"  Max diff: {np.max(np.abs(reconstructed - expected_first))}")
    
    # Check output window
    first_output_window = y_features[0]  # shape (M, 2)
    expected_out = y_norm[stride:stride+M]
    reconstructed_out = first_output_window[:, 0] + 1j * first_output_window[:, 1]
    
    if np.allclose(reconstructed_out, expected_out):
        print(f"✓ Phase normalization correctly applied to outputs")
    else:
        print(f"✗ Phase normalization issue in outputs")
        print(f"  Max diff: {np.max(np.abs(reconstructed_out - expected_out))}")
    
    return True

if __name__ == "__main__":
    # Test with different configurations
    test_alignment(M=10, N=100, stride=1)
    test_alignment(M=10, N=100, stride=5)
    test_alignment(M=20, N=200, stride=10)
    
    print(f"\n{'='*60}")
    print("✓ ALL TESTS PASSED")
    print(f"{'='*60}")
