from Volterra import Volterra
from Dataset import Dataset
from Datapath import Datapath
from DataManager import DataManager
from NeuralNetwork import NeuralNetwork, PNTDNN, PNTDNN_3_layers

dataManager = DataManager(filepath='C:\\Users\\AQUIGNEY\\Documents\\College\\Project\\PA_IO.mat', num_training_points=10000, num_validaiton_points=2000, num_test_points=2000)


# Setup a volterra model and a volterra inverse
volterra_forward_model = Volterra(num_nl_orders=5, num_memory_levels=3, dataset=dataManager.training_dataset)
volterra_invserse_model = Volterra(num_nl_orders=5, num_memory_levels=3, dataset=dataManager.training_dataset)

training_dataset = dataManager.training_dataset

# Traing small Inverse Model 
nn_inv = NeuralNetwork(num_memory_levels=28, model_type='PNTDNN', forward_model=False)
best_nn_model = nn_inv.get_best_model(num_epochs=200, training_dataset=training_dataset, validation_dataset=dataManager.validation_dataset, learning_rate=1e-3)

# Create datapath with volterra forward model and nn inverse model
datapath = Datapath(forward_model=volterra_forward_model, inverse_model=nn_inv)
# Process test data through datapath
test_dataset = datapath.process(dataManager.test_dataset.input_data)
# Calculate NMSE
nmse = test_dataset.calculate_nmse()
print(f"Test NMSE: {nmse:.2f} dB")
# Plot input vs output magnitude
Datapath.plot_signals(test_dataset)