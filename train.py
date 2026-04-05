# Train model based on input output file
from sparseDPD import PNTDNN_NeuralNetwork, DataManager, PhysicalDatapath, Dataset


# Read input output (IQ data) from file
# Same dataset/training style as refactored_notebook.ipynb
openDPD_folder = 'OpenDPD_datasets/DPA_160MHz'

#40,000 points
openDPD_dataManager = DataManager(
    num_test_points=98304,
    num_training_points=294912,
    num_validaiton_points=98304,
    openDPD_test_input_file=f'{openDPD_folder}/test_input.csv',
    openDPD_test_output_file=f'{openDPD_folder}/test_output.csv',
    openDPD_training_input_file=f'{openDPD_folder}/train_input.csv',
    openDPD_training_output_file=f'{openDPD_folder}/train_output.csv',
    openDPD_validation_input_file=f'{openDPD_folder}/val_input.csv',
    openDPD_validation_output_file=f'{openDPD_folder}/val_output.csv'
)

# Instantiate Train PNTDNN model
PNTDNN_inverse_nn = PNTDNN_NeuralNetwork(num_memory_levels=15, model_type='OneLayerNetwork', nn_file_path=None, forward_model=False)
physical_datapath = PhysicalDatapath(inverse_model=PNTDNN_inverse_nn)


# Build a data set using the input from openDPD, and the output from the PA
training_input = openDPD_dataManager.training_dataset.input_data
training_output = physical_datapath.send_to_physical_pa(training_input)
training_dataset = Dataset(input_data=training_input, output_data=training_output)

valid_input = openDPD_dataManager.validation_dataset.input_data
valid_output = physical_datapath.send_to_physical_pa(valid_input)
valid_dataset = Dataset(input_data=valid_input, output_data=valid_output)

# Now use actual PA as forward model

physical_datapath.train_using_ila(training_dataset=training_dataset, valid_dataset=valid_dataset, iterations=5, retrain_epochs_per_iteration=50)


# Measure PA output with the trained inverse model as input to the PA
test_input = openDPD_dataManager.test_dataset.input_data
test_output = physical_datapath.send_to_physical_pa(test_input)
test_dataset = Dataset(input_data=test_input, output_data=test_output)

nmse = test_dataset.calculate_nmse()
print(f"Final NMSE on test set: {nmse:.4f} dB")