# FedTraffic-RL: DQN-Based Traffic Light Control

This project implements a Deep Q-Network (DQN) agent to optimize traffic light timings at a four-way intersection using the **SUMO (Simulation of Urban MObility)** environment. The agent aims to maximize traffic flow while minimizing congestion and waiting times through adaptive phase switching.

## 🚀 Key Features

- **Deep Q-Learning:** Implements a DQN agent with Experience Replay and a Target Network for stable training.
- **Centralized Configuration:** All hyperparameters and simulation constants are managed in a single `src/config.py` file.
- **Adaptive Phase Control:** The agent controls an 8-phase traffic light system (Green and Yellow phases for North, East, South, and West).
- **Deterministic Traffic Scenarios:**
  - `burst_spawn`: A high-volume surge of vehicles at the start of the simulation to test clearance efficiency.
  - `periodic_uniform`: A steady, regular flow of vehicles arriving at fixed intervals to test sustained throughput.
- **Advanced Reward Logic:** A multi-objective reward function that balances junction throughput (flow rate) against local congestion (halting cars).

## 🛠️ Project Structure

- `src/config.py`: Centralized hub for hyperparameters (LR, Gamma, Batch Size) and simulation settings (Step length, Reward weights).
- `src/agent.py`: Contains the `TrafficLightDQN` architecture and the `TrafficLightAgent` logic.
- `src/control.py`: The main training script. Orchestrates the simulation loop and agent learning.
- `src/evaluate.py`: Comprehensive evaluation tool that compares the trained agent against a fixed-time baseline.
- `src/generate_routes.py`: Generates SUMO `.rou.xml` files for deterministic traffic spawning.
- `sumo_config/`: Contains the network definitions (`.net.xml`) and simulation configuration (`.sumocfg`).

## 🧠 Reward Function

The agent's behavior is driven by the following reward formulation:
$$Reward = (W_1 \times \text{Flow Rate}) - (W_2 \times \text{Waiting Cars})$$

- **Flow Rate:** The number of vehicles successfully entering outgoing edges during a step.
- **Waiting Cars:** A penalty based on the total number of halting vehicles across all incoming lanes.
- **Weights:** $W_1$ and $W_2$ are adjustable in `config.py` to prioritize either throughput or queue reduction.

## 🚦 Getting Started

### Prerequisites

- Python 3.8+
- [SUMO](https://www.eclipse.org/sumo/) (ensure `SUMO_HOME` is set in your environment variables).
- PyTorch, NumPy, and TraCI.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Project

**1. Training the Agent**
Train the agent on a specific traffic scenario (e.g., `burst_spawn`):

```bash
python src/control.py --experiment burst_spawn
```

**2. Evaluating Performance**
Compare your trained model against a fixed-time baseline. This generates detailed analysis plots in the `results/` directory:

```bash
python src/evaluate_new.py --experiment burst_spawn
```

## 📊 Visualizations

The evaluation script generates four distinct performance metrics:

1.  **Summary Evaluation**: A dual-plot showing cumulative reward and smoothed queue lengths.
2.  **Cumulative Wait Time**: Tracks the total accumulated delay over the simulation duration.
3.  **Wait Time Distribution**: A boxplot showing the spread of wait times for all completed vehicles.
4.  **Phase Timeline**: A visual breakdown of when the agent chose to switch lights compared to the baseline.
