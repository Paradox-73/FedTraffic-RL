# Federated Learning for Robotics Traffic Control

This project implements an intelligent traffic light control system using Deep Reinforcement Learning (DQN). It progresses from a single-agent baseline to complex traffic scenarios (Rush Hour, Gridlock) and is designed to evolve into a **Federated Learning** system where multiple intersections learn collaboratively without sharing raw data.

## 📂 Project Structure

```text
FedLearning-Robotics/
├── src/
│   ├── agent.py          # The DQN Agent (Neural Network, Memory, Learning Logic)
│   ├── control.py        # Main Training Loop (SUMO interaction, Route generation)
│   ├── evaluate.py       # Evaluation Script (Loads trained models for testing)
│   └── client.py         # [Upcoming] Flower Client for Federated Learning
├── sumo_config/
│   ├── hello.net.xml     # Network definition (Junctions, Roads, Lanes)
│   ├── hello.rou.xml     # Dynamic Route file (Generated per episode)
│   ├── hello.sumocfg     # SUMO Configuration wrapper
│   └── stageX/           # Specific configs for different stages
├── models/
│   ├── stage1_baseline/  # Saved .pth models for Stage 1
│   ├── stage2_rush_hour/ # Saved .pth models for Stage 2
│   └── stage3_gridlock/  # Saved .pth models for Stage 3
├── results/
│   └── ...               # Training plots (Reward/Loss curves)
├── logs/
│   └── ...               # Text logs of simulation runs
└── requirements.txt      # Python dependencies
```

## 🏗️ System Architecture

1.  **The Environment (SUMO)**

    **Physics:**
    Vehicles have distinct characteristics (Cars, Buses, Motorcycles) and driver imperfection (speed variance).

    **State Space (15-dim):**
    The agent sees the intersection through:
    - Queue Lengths: Number of halted vehicles on N, S, E, W arms.
    - Waiting Times: Cumulative wait time per arm.
    - Pressure: The density difference (Incoming - Outgoing vehicles).
    - Phase Info: Current phase ID and time elapsed.

    **Action Space (Discrete):**
    - 0: Keep current phase.
    - 1: Switch phase.

2.  **The Agent (DQN)**

    The "Brain" is a Deep Q-Network implemented in PyTorch.

    **Experience Replay:**
    Stores memories (state, action, reward, next_state) to break correlation and stabilize training.

    **Target Network:**
    Uses a separate network for target value calculation to prevent oscillation.

    **Huber Loss:**
    Robust to outliers (e.g., sudden massive traffic jams).

## 📊 Experimental Stages & Results

We trained the agent on three distinct "levels" of difficulty.

**Stage 1: Baseline (Standard Intersection)**

- **Scenario:** Moderate, balanced traffic with random volume fluctuations (+/- 10%).
- **Goal:** Learn basic queue clearing logic.
- **Result:**
  Interpretation: The Blue Line (Reward) shows a steady upward trend, indicating the agent learned to reduce waiting times. The Red Line (Loss) decreases rapidly, showing the neural network successfully converged on a strategy.

**Stage 2: Rush Hour (Non-Stationary)**

- **Scenario:** A realistic morning commute simulation.
  - 0-300s: Lull (Low Traffic).
  - 300-700s: Spike (Heavy North-South flow).
  - 700s+: Cooldown.
- **Goal:** Adaptation. The agent must switch strategies mid-episode to prioritize the main artery.
- **Result:**
  Interpretation: You can see a dip in rewards corresponding to the traffic spike, but the agent recovers. The consistent low loss indicates it successfully generalized the concept of "prioritization."

**Stage 3: Gridlock (High Conflict)**

- **Scenario:** Heavy traffic with Blocking Left Turns enabled on a single-lane road.
- **Goal:** Geometry Management. Left-turning cars block the lane while yielding, causing "phantom jams."
- **Result:**
  Interpretation: This graph is more volatile (spiky) because physical blockages are unpredictable. However, the upward trend in reward proves the agent learned non-linear strategies to clear these blockages.

## 📈 Evaluation Performance

After training for 100 episodes, we ran the agent in "Exploitation Mode" (No random actions). The metric is Total Accumulated Reward (Net change in wait time). A score close to 0 is optimal.

| Experiment          | Score | Verdict                                         |
| :------------------ | :---- | :---------------------------------------------- |
| Stage 1 (Baseline)  | -0.23 | ✅ Excellent stability.                         |
| Stage 2 (Rush Hour) | -0.15 | 🏆 Best Performance (Cleared queues perfectly). |
| Stage 3 (Gridlock)  | -0.23 | ✅ Successfully prevented deadlock.             |

## 🚀 Phase 3: Federated Learning (Next Steps)

Now that we have a capable local agent, we are moving to Federated Learning using the Flower (flwr) framework.

**Why Federated Learning?**
In the real world, cities cannot upload raw camera feeds from every intersection to a central cloud due to Privacy and Bandwidth constraints.

**The Architecture**

- **Local Training (Clients):** Each intersection (Client) trains its own local model on its own traffic data.
- **Aggregation (Server):** Clients send only their Model Weights (not data) to the server.
- **Federated Averaging:** The server averages the weights to create a "Global Smart Model" and sends it back to all clients.

**Implementation Plan**

- `src/client.py`: Wrap the existing TrafficLightAgent into a `flwr.client.NumPyClient`.
- `src/server.py`: Create a `flwr.server` to orchestrate the rounds.
- Multi-Agent Simulation: Spin up 3 parallel SUMO instances (representing 3 different intersections) and train them simultaneously.

## 🛠️ How to Run

**Prerequisites**

- SUMO installed and `SUMO_HOME` environment variable set.
- Python 3.8+

**Dependencies:**

- `pip install -r requirements.txt`

1.  **Training**
    Run the control script to start training. You can specify the experiment stage.

    ```bash
    # Run the Baseline
    python src/control.py --experiment stage1_baseline

    # Run the Rush Hour scenario
    python src/control.py --experiment stage2_rush_hour

    # Run ALL experiments sequentially
    python src/control.py --experiment all
    ```

2.  **Evaluation**
    To test the trained models:
    ```bash
    python src/evaluate.py --experiment stage1_baseline --nogui
    python src/evaluate.py --experiment stage2_rush_hour --nogui
    python src/evaluate.py --experiment stage3_gridlock --nogui
    ```
