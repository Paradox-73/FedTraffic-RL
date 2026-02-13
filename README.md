# Federated Learning for Robotics Traffic Control

This project implements an intelligent traffic light control system using Deep Reinforcement Learning (DQN). It progresses from a single-agent baseline to complex traffic scenarios (Rush Hour, Gridlock) and is designed to evolve into a **Federated Learning** system where multiple intersections learn collaboratively without sharing raw data.

## 📂 Project Structure

```text
FedLearning-Robotics/
├── src/
│   ├── agent.py          # The DQN Agent (Neural Network, Memory, Learning Logic)
│   ├── control.py        # Main Training Loop (SUMO interaction, Route generation)
│   ├── evaluate.py       # Evaluation Script (Compares trained models to a baseline)
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
│   └── ...               # Training & Evaluation plots
├── logs/
│   └── ...               # Text logs of simulation runs
└── requirements.txt      # Python dependencies
```

## 🏗️ System Architecture

1.  **The Environment (SUMO)**

    **Physics:**
    Vehicles have distinct characteristics (Cars, Buses, Motorcycles) and driver imperfection (speed variance).

    **State Space (18-dim):**
    The agent sees the intersection through:
    - **Queue Lengths (4):** Number of halted vehicles on N, S, E, W arms.
    - **Waiting Times (4):** Cumulative wait time per arm.
    - **Phase One-Hot (4):** Which direction is green? `[N, E, S, W]`
    - **Time Elapsed (1):** How long the current green phase has been active.
    - **Pressure (4):** The density difference (Incoming - Outgoing vehicles) per arm.
    - **Bias (1):** A constant bias term.

    **Action Space (Discrete):**
    - 0: Keep current green phase.
    - 1: Switch to the next phase (initiating a yellow light).

2.  **The Agent (DQN)**

    The "Brain" is a Deep Q-Network implemented in PyTorch.

    **Experience Replay:**
    Stores memories (state, action, reward, next_state) to break correlation and stabilize training.

    **Target Network:**
    Uses a separate network for target value calculation to prevent oscillation.

    **Huber Loss:**
    Robust to outliers (e.g., sudden massive traffic jams).

## 📊 Experimental Stages

We train the agent on three distinct "levels" of difficulty.

**Stage 1: Baseline (Standard Intersection)**

- **Scenario:** Moderate, balanced traffic with random volume fluctuations.
- **Goal:** Learn basic queue clearing logic.

**Stage 2: Rush Hour (Non-Stationary)**

- **Scenario:** A realistic morning commute with a heavy North-South traffic spike mid-episode.
- **Goal:** Adapt strategies mid-episode to prioritize the main artery.

**Stage 3: Gridlock (High Conflict)**

- **Scenario:** Heavy traffic with single-lane roads where left-turning cars can block the lane, causing "phantom jams."
- **Goal:** Learn to manage and clear unpredictable physical blockages.

## 🚀 Federated Learning (Next Steps)

Now that we have a capable local agent, the project is designed to expand into a Federated Learning system using the Flower (`flwr`) framework.

**Why Federated Learning?**
In the real world, cities cannot upload raw camera feeds from every intersection to a central cloud due to Privacy and Bandwidth constraints. Federated Learning allows multiple intersections to build a smarter, collaborative model by only sharing their anonymous model weights, not sensitive traffic data.

**Implementation Plan**

- `src/client.py`: Wrap the existing `TrafficLightAgent` into a `flwr.client.NumPyClient`.
- `src/server.py`: Create a `flwr.server` to orchestrate the training rounds.
- Multi-Agent Simulation: Spin up multiple parallel SUMO instances and train them simultaneously as a federation.

## 🛠️ How to Run

**Prerequisites**

- SUMO installed and the `SUMO_HOME` environment variable set.
- Python 3.8+

**Dependencies:**

```bash
pip install -r requirements.txt
```

### 1. Training

Run the control script to start training. You must specify the experiment stage. Use the `--nogui` flag to run without the SUMO GUI for faster training.

```bash
# Run the Baseline scenario (with GUI)
python src/control.py --experiment stage1_baseline

# Run the Rush Hour scenario (headless)
python src/control.py --experiment stage2_rush_hour --nogui

# Run ALL experiments sequentially (headless)
python src/control.py --experiment all --nogui
```

Training will save model files to `models/<experiment_name>/` and learning curve plots to `results/<experiment_name>/`.

### 2. Evaluation

After training, you can evaluate the models using `evaluate.py`. This script compares the performance of trained RL models against a fixed-time baseline and generates plots.

**Usage:**

```bash
python src/evaluate.py --help
```

**Examples:**

- **Evaluate all experiments (default behavior):**
  This will find all trained models in `models/` and generate comparison plots for each.

  ```bash
  python src/evaluate.py
  ```

- **Evaluate a specific experiment:**

  ```bash
  python src/evaluate.py --experiment stage1_baseline
  ```

- **Evaluate a specific experiment and specify a custom model file:**
  (Note: The `--model` argument is only used when `--experiment` specifies a single experiment, not 'all'.)

  ```bash
  python src/evaluate.py --experiment stage1_baseline --model models/stage1_baseline/my_custom_model.pth
  ```

- **Evaluate a specific experiment and add a suffix to the plot filename:**
  This is useful for comparing different models or evaluation runs without overwriting previous plots. The plot will be saved as `comparison_plot_mytest.png`.

  ```bash
  python src/evaluate.py --experiment stage1_baseline --suffix _mytest
  ```

- **Evaluate with SUMO GUI visible (for debugging):**
  ```bash
  python src/evaluate.py --experiment stage1_baseline --nogui
  ```
  (Note: The `--nogui` flag in `evaluate.py` controls the SUMO GUI visibility during evaluation simulations.)

Evaluation plots will be saved to `results/<experiment_name>/comparison_plot<suffix>.png`.
