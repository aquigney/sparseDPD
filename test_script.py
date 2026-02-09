# Quick test script, delete later
from sparseDPD import Volterra
from sparseDPD import Dataset
from sparseDPD import Datapath
from sparseDPD import DataManager
from sparseDPD import NeuralNetwork
from sparseDPD import DeltaGRUNetwork
from sparseDPD import Experiment, LinearExperiment


simpleDataManager = DataManager(filepath='PA_IO.mat', num_training_points=5000, num_validaiton_points=5000, num_test_points=2000)

# Setup a volterra model and a volterra inverse
volterra_forward_model = Volterra(num_nl_orders=5, num_memory_levels=3, dataset=simpleDataManager.training_dataset)

print(f" Volterra NMSE: {volterra_forward_model.calculate_volterra_nmse(simpleDataManager.test_dataset)} dB")

# Test NN forward performance
forward_nn = NeuralNetwork(num_memory_levels=7, model_type='PNTDNN', forward_model=True)
train_losses_fwd, valid_losses_fwd, best_epoch_fwd = forward_nn.get_best_model(num_epochs=10, training_dataset=simpleDataManager.training_dataset, validation_dataset=simpleDataManager.validation_dataset, learning_rate=1e-2)
# Print NMSE for forward model
fwd_nmse = forward_nn.calculate_forward_nmse(simpleDataManager.test_dataset)
print(f"NN Forward Model NMSE: {fwd_nmse:.2f} dB")


# Create a Linear Pruning Experiment
exp = LinearExperiment(forward_nn, 1, 0.1, 50, simpleDataManager.get_training_data(), simpleDataManager.get_validation_data(), simpleDataManager.get_test_data())

exp.run()