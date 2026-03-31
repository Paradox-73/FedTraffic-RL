import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Simulation Constants ---
SIMULATION_TIME = 500  # Total steps per episode
NUM_OF_EPISODES = 200
MIN_GREEN_TIME = 10
YELLOW_TIME = 3
ACTION_COMMIT_TIME = 10
BASELINE_GREEN_TIME = 30

# --- Spawning Configuration ---
BURST_COUNT = 5
PERIOD = 12
HEAVY_PERIOD = 2   # High traffic
LIGHT_PERIOD = 30  # Low traffic

# --- DQN Hyperparameters ---
LEARNING_RATE = 5e-5
GAMMA = 0.98
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# --- Reward Weights ---
# Throughput weight (cars passed) - Increased to provide stronger signal
W1 = 20.0
W2 = 1.0   # Congestion penalty weight (halting cars)

# --- Traffic Phase Definitions ---
PHASE_N_GREEN = 0
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4
PHASE_S_YELLOW = 5
PHASE_W_GREEN = 6
PHASE_W_YELLOW = 7

GREEN_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN, PHASE_S_GREEN, PHASE_W_GREEN]
GREEN_PHASES_NAME = ["North", "East", "South", "West"]
YELLOW_PHASES = {
    PHASE_N_GREEN: PHASE_N_YELLOW,
    PHASE_E_GREEN: PHASE_E_YELLOW,
    PHASE_S_GREEN: PHASE_S_YELLOW,
    PHASE_W_GREEN: PHASE_W_YELLOW,
}

INCOMING_EDGES = ["edge_N_in", "edge_S_in", "edge_E_in", "edge_W_in"]
OUTGOING_EDGES = ["edge_N_out", "edge_S_out", "edge_E_out", "edge_W_out"]
