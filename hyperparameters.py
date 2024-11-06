# path where al results will be logged
path = 'best_results/'

# circuit parameters
NUM_QUBITS   = 3
NUM_LARGE    = 20
NUM_SMALL1   = 15
NUM_SMALL2   = 15

# training parameters
LEARNING_RATE      = 5e-4
BATCH_SIZE         = 64
NUM_EPOCHS         = 100
SCHEDULER_PATIENCE = 20
SCHEDULER_GAMMA    = 0.7
DATA_LENGTH        = 2048

# diffusion model parameters
T     = 8
beta0 = 4e-3
betaT = 4e-1

# Autoencoder parameters
LATENT_DIM = 8
