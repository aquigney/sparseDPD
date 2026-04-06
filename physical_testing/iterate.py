# Run iterative training. This file should take in a list of input and output data (as a .mat)
# And use this to train a baseline model
from sparseDPD import PNTDNN_NeuralNetwork, DataManager, PhysicalDatapath, Dataset

openDPD_folder = 'OpenDPD_datasets/DPA_160MHz'

# Load in data here
simpleDataManager = DataManager(
    num_training_points=20000,
    num_validaiton_points=2000,
    num_test_points=2000,
    filepath='UCD_datasets/PA_IO.mat'
)

# Read the current NN file
PNTDNN_inverse_nn = PNTDNN_NeuralNetwork(num_memory_levels=15, model_type='OneLayerNetwork', nn_file_path='physical_testing/pntdnn_inverse_model_open.pt', forward_model=False)

# Train this model for 50 epochs on the training data.
PNTDNN_inverse_nn.get_best_model(num_epochs=50, training_dataset=simpleDataManager.training_dataset, validation_dataset=simpleDataManager.validation_dataset)

# Process output data and output to .mat file
output_data = PNTDNN_inverse_nn.generate_model_output(simpleDataManager.test_dataset.input_data)

# Save input and output to a file, in the same structure as the original .mat file
simpleDataManager.save_to_mat_file(output_data, 'physical_testing/pntdnn_inverse_output.mat')

# Save the new PNTDNN model to a file (iterate number at the end, or put 1 is there is none)
PNTDNN_inverse_nn.write_nn_to_file('physical_testing/pntdnn_inverse_model_open.pt')