# Generate a Volterra Series Object
import numpy as np

class Volterra:
    def __init__(self, num_nl_orders, num_memory_levels, dataset, forward=True):
        """Class to create a Volterra model"""
        self.num_nl_orders = num_nl_orders
        self.num_memory_levels = num_memory_levels
        self.dataset = dataset
        input_data = dataset.input_data
        output_data = dataset.output_data
        self.forward = forward

        self.A = self.build_coeff_matrix(input_data, output_data) # Coefficient matrix

    def build_x_matrix(self, x):
        """Builds a Volterra X matrix for input signal x"""
        num_points = len(x)
        X = np.zeros((num_points, self.num_memory_levels * self.num_nl_orders), dtype=np.complex128)
        
        for n in range(self.num_memory_levels - 1, num_points):
            col = 0
            for i in range(self.num_memory_levels):
                xi = x[n - i]
                for j in range(self.num_nl_orders):
                    X[n, col] = (abs(xi) ** ((j) * 2)) * xi
                    col += 1

        return X
    
    def build_y(self, u):
        """Build the output vector of the Volterra Model"""
        if self.A is not None:
            num_points = len(u)
            y = np.zeros((num_points,), dtype=np.complex128)
            for n in range(self.num_memory_levels - 1, num_points):
                col = 0 
                for i in range(self.num_memory_levels):
                    ui = u[n-i]
                    for j in range(self.num_nl_orders):
                        y[n]= y[n] + self.A[col]*(abs(ui)**(j*2)*ui)
                        col += 1
            y = y[self.num_memory_levels:]  # Use parameter, not self.num_memory_levels
            return y
        else:
            print("Coefficient matrix A is not defined.")
            return None 
        
    def build_coeff_matrix(self, input, output, check_conditioning=False):
        """Build the Volterra Coefficient Matrix A using Least Squares. 
            Check if the matrix becomes ill-conditioned if specified.
        """

        X = self.build_x_matrix(input)
        X_trimmed = X[self.num_memory_levels:, :]
        y_trimmed = output[self.num_memory_levels:]

        gram_matrix = X_trimmed.conj().T @ X_trimmed
        if check_conditioning:
            cond_number = np.linalg.cond(gram_matrix)
            if cond_number > 1e12:
                print(f"Warning: The Gram matrix is ill-conditioned with condition number {cond_number}")

        self.A = np.linalg.pinv(gram_matrix) @ X_trimmed.conj().T @ y_trimmed
        return self.A
    
    
    def calculate_volterra_nmse(self, dataset):
        """Calculate how closely the volterra model fits the actual PA output"""
        input = dataset.input_data # Input data
        output = dataset.output_data # Actual PA output data

        volterra_output = self.build_y(input).squeeze()
        actual_output = output[self.num_memory_levels:].squeeze()

        #Ensure both arrays are of the same length
        min_length = min(len(volterra_output), len(actual_output))
        volterra_output = volterra_output[:min_length]
        actual_output = actual_output[:min_length]

        # Calculate NMSE
        error_power = np.mean(np.abs(volterra_output - actual_output)**2)
        signal_power = np.mean(np.abs(actual_output)**2)
        nmse = error_power / signal_power
        nmse_db = 10 * np.log10(nmse)
        
        return nmse_db