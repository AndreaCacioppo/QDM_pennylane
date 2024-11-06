import torch
import pennylane as qml

from hyperparameters import *

NUM_QUBITS += 4

# get the device
Bdev = qml.device("default.qubit", wires=NUM_QUBITS-1)
Sdev = qml.device("default.qubit", wires=NUM_QUBITS)
Ldev = qml.device("default.qubit", wires=NUM_QUBITS+1)

@qml.qnode(Bdev, interface="torch")
def Bblock(params, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS-1))
    
    qml.StronglyEntanglingLayers(params, wires=range(NUM_QUBITS-1), ranges = [1]*params.shape[0])

    # Return the state vector
    return qml.state()

@qml.qnode(Sdev, interface="torch")
def Sblock(params, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))
    
    qml.StronglyEntanglingLayers(params, wires=range(NUM_QUBITS), ranges = [1]*params.shape[0])

    # Return the state vector
    return qml.state()

@qml.qnode(Ldev, interface="torch")
def Lblock(params, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS+1))

    qml.StronglyEntanglingLayers(params, wires=range(NUM_QUBITS+1), ranges = [1]*params.shape[0])
    
    # Return the state vector
    return qml.state()

# define general circuit
def circuit(S1params, Lparams, S2params, state = None, labels_batch = None):

    # extend the state by taking the tensor product with the ancilla that encodes the labels
    state = extend_state(state, labels_batch)

    # apply first small block
    state = Sblock(S1params, state)

    # take the tensor product to extend the state: |0>*|state>
    ext_state = torch.cat((state, torch.zeros_like(state)), dim = 1)

    # apply large block
    ext_state = Lblock(Lparams, ext_state)

    ##########################################

    # Implement random measurement instead of post selection
    # Calculate the probability of having label 0
    prob0 = torch.norm(ext_state[:, :2**NUM_QUBITS], dim = 1)**2

    # extrat between 0 and 1 with shape prob0.shape
    random_batch = torch.rand_like(prob0)

    # where random_batch < prob0, set measure_batch to the first half of the state, otherwise set it to the second half
    indexes = torch.where(random_batch < prob0, torch.zeros_like(random_batch), torch.ones_like(random_batch))

    # Calculate actual start indices
    start_indices = indexes * 2**NUM_QUBITS

    # Create a range tensor of [0,1,2,3,...16]
    range_tensor = torch.arange(2**NUM_QUBITS).unsqueeze(0).to(state.device)

    # Expand start_indices from shape [64] to shape [64, 16]
    expanded_start_indices = start_indices.unsqueeze(1).expand(-1, 2**NUM_QUBITS)

    # Add range_tensor to each row of expanded_start_indices
    index_tensor = expanded_start_indices.long() + range_tensor

    # Use advanced indexing to get the slices
    state = ext_state[torch.arange(state.shape[0]).unsqueeze(1), index_tensor]

    ##########################################

    # renormalise
    state = state/torch.linalg.vector_norm(state, dim = 1).view(-1, 1)

    # apply second small block
    state = Sblock(S2params, state)

    # measure ancilla and accept only the right ones depending on the labels
    state = reduce_state(state, labels_batch)

    return state

# define autoencoder
def auto_encoder(S1params, Bparams, S2params, state = None, labels_batch = None):

    # extend the state by taking the tensor product with the ancilla
    state = extend_state(state, labels_batch)

    # apply first small block
    state = Sblock(S1params, state)

    ##########################################

    # Implement random measurement instead of post selection
    # Calculate the probability of having label 0
    prob0 = torch.norm(state[:, :2**NUM_QUBITS], dim = 1)**2

    # extrat between 0 and 1 with shape prob0.shape
    random_batch = torch.rand_like(prob0)

    # where random_batch < prob0, set measure_batch to the first half of the state, otherwise set it to the second half
    indexes = torch.where(random_batch < prob0, torch.zeros_like(random_batch), torch.ones_like(random_batch))

    # Calculate actual start indices
    start_indices = indexes * 2**NUM_QUBITS

    # Create a range tensor of [0,1,2,3,...16]
    range_tensor = torch.arange(2**NUM_QUBITS).unsqueeze(0)

    # Expand start_indices from shape [64] to shape [64, 16]
    expanded_start_indices = start_indices.unsqueeze(1).expand(-1, 2**NUM_QUBITS)

    # Add range_tensor to each row of expanded_start_indices
    index_tensor = expanded_start_indices.long() + range_tensor

    # Use advanced indexing to get the slices
    state = state[torch.arange(state.shape[0]).unsqueeze(1), index_tensor]

    ##########################################

    # renormalise
    state = state/torch.linalg.vector_norm(state, dim = 1).view(-1, 1)

    # apply latent block
    state = Bblock(Bparams, state)

    # add |0> qubit
    state = torch.cat((state, torch.zeros_like(state)), dim = 1)

    # apply second small block
    state = Sblock(S2params, state)

    # mesdure ancilla and accept only the right ones depending on the labels
    state = reduce_state(state, labels_batch)

    return state

# enrich state with ancilla qubits encoding labels
def extend_state(state_batch, labels_batch):

    extended_state = torch.zeros((state_batch.shape[0], 2**NUM_QUBITS), dtype=state_batch.dtype).to(state_batch.device)

    indices = (labels_batch * 8).view(-1, 1) + torch.arange(8).view(1, -1).to(labels_batch.device)

    # fill the extended state with state_batch in a position depending on labels_batch
    # this is equivalent to taking the tensor product
    extended_state.scatter_(1, indices, state_batch)

    return extended_state

# reduce state by measruing ancilla qubit depending on labels and renormalise
def reduce_state(state, labels_batch):

    # Calculate actual start indices
    start_indices = labels_batch * 8

    # Create a range tensor of [0,1,2,3,...7]
    range_tensor = torch.arange(8).unsqueeze(0).to(state.device)

    # Expand start_indices from shape [64] to shape [64, 8]
    expanded_start_indices = start_indices.unsqueeze(1).expand(-1, 8)

    # Add range_tensor to each row of expanded_start_indices
    index_tensor = expanded_start_indices + range_tensor

    # Use advanced indexing to get the slices
    reduced_state = state[torch.arange(state.shape[0]).unsqueeze(1), index_tensor]

    # normalise along the second axis
    reduced_state = reduced_state/torch.linalg.vector_norm(reduced_state, dim = 1).view(-1, 1)

    return reduced_state

'''# Implement random measurement instead of post selection

# Calculate the probability of having each label 
state = state.view(state.shape[0], 2**4, 2**3)
probs = torch.norm(state, dim = 2)**2
state = state.view(state.shape[0], -1)

# make a cumulative sum of the probabilities
cum_probs = torch.cumsum(probs, dim = 1)

# extract batch of 64 random numbers
random_batch = torch.rand(state.shape[0]).unsqueeze(1)

# implement measurement
measurements = torch.sum(cum_probs <= random_batch, dim=1)

# Create a tensor of shape (64, 16) to store the start indices for slicing
start_indices = measurements.unsqueeze(1) * 8

# Generate indices for slicing along the second dimension
indices = torch.arange(8)

# Expand dimensions for broadcasting
indices = indices.unsqueeze(0)

# Calculate the final indices for slicing
sliced_indices = start_indices + indices

# Assuming you have a tensor 'data' with shape (64, 128) that you want to slice
reduced_state = state.gather(1, sliced_indices)'''
