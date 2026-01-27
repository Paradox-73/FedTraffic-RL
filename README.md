# Federated Reinforcement Learning for Adaptive Traffic Signal Control

This project implements a single, intelligent traffic light agent that learns to optimize traffic flow for a 4-way intersection using Deep Reinforcement Learning (DQN). It serves as the foundational step for a larger system designed to use Federated Learning (via Flower) to enable multiple intersections to learn collaboratively.

## 1. System Architecture
The system consists of two main components that interact in a classic agent-environment loop:
-   **The Environment (SUMO):** A realistic traffic simulator that provides the state of the intersection and calculates the reward signal.
-   **The Agent (Python):** A DQN-based agent that perceives the state, makes decisions, and learns from the consequences of its actions.

---

## 2. The SUMO Simulation Environment
The simulation environment, configured in the `sumo_config/` directory, models the physical world for our agent.

-   `hello.net.xml`: Defines the physical road network, which consists of a single 4-way intersection (`J1`).
-   `hello.rou.xml`: Defines the traffic flow. It specifies routes for different types of vehicles (cars, buses, motorcycles) that are spawned at regular intervals.

### How is traffic simulated?
Traffic is generated based on the routes and vehicle definitions in `hello.rou.xml`. Each vehicle follows its predetermined path. The agent's only influence over the simulation is its ability to change the traffic light phases at intersection `J1`.

### Will tinkering with traffic change the results?
**Yes, absolutely.** The policy learned by the agent is highly specific to the traffic patterns it was trained on. If you modify `hello.rou.xml` to create more or less traffic, or change the routes vehicles take, the pre-trained model's performance will degrade. It would need to be retrained on the new traffic patterns to learn an effective new policy.

---

## 3. The Reinforcement Learning Agent
The "brain" of the traffic light is defined in `src/agent.py`. It uses a Deep Q-Network (DQN), a popular model-free reinforcement learning algorithm.

### State Space (What the Agent Sees)
To make a decision, the agent perceives the environment as a 12-dimensional state vector:
-   **Elements 0-3**: Queue lengths (number of halting vehicles) on the North, South, East, and West approaches.
-   **Elements 4-7**: Average vehicle waiting time on the N, S, E, W approaches.
-   **Element 8**: The current phase of the traffic light (1 if North/South is green, 0 otherwise).
-   **Element 9**: The time elapsed (in simulation steps) since the current green phase began.
-   **Element 10**: A flag indicating if the light is currently yellow (1 for yes, 0 for no).
-   **Element 11**: A constant bias term (always 1.0).

### Action Space (What the Agent Does)
The agent has a simple choice at each decision point:
-   `0`: **Keep** the current traffic light phase.
-   `1`: **Switch** to the next phase in the cycle.

### Reward Function (What the Agent Wants)
The agent's goal is to maximize its cumulative reward. The reward is defined as the **negative total waiting time** of all vehicles in the simulation.
-   **Why is it negative?** By framing the goal as maximizing a negative number, the agent is incentivized to get that number as close to zero as possible. Maximizing `-(waiting_time)` is equivalent to minimizing `waiting_time`.

### RL Best Practices and Implementation
To ensure the agent learns effectively and stably, several key DQN enhancements were implemented in `src/agent.py`:

1.  **Experience Replay:** The agent stores its experiences `(state, action, reward, next_state)` in a memory buffer. During training, it samples random mini-batches from this buffer. This breaks the correlation between consecutive experiences, leading to more stable and robust learning.
2.  **Target Network:** Two neural networks are used: a primary model and a "target" model. The primary model is constantly learning, while the target model's weights are only updated periodically. This provides a stable target for the loss calculation, preventing the agent from "chasing a moving target" and helping to prevent feedback loops during training.
3.  **Huber Loss (`SmoothL1Loss`):** Instead of Mean Squared Error (MSE), Huber Loss is used. This loss function acts like MSE for small errors but like Mean Absolute Error (MAE) for large errors. It is much less sensitive to the occasional very large prediction errors (outliers) that are common in RL, preventing unstable updates and exploding gradients.
4.  **Gradient Clipping:** After the loss is calculated, the gradients are "clipped" to a maximum norm of 1.0. This acts as a final safeguard against exploding gradients, further ensuring the stability of the learning process.

---

## 4. How to Use This Project

### Prerequisites
1.  Install **SUMO** and set the `SUMO_HOME` environment variable.
2.  Install Python dependencies: `pip install -r requirements.txt`

### Step 1: Train a New Agent
To start the training process, run `control.py`. This will launch the SUMO GUI and begin the 200-episode training loop. All terminal output will be saved to a new file in the `logs/` directory.
```bash
python src/control.py
```

### Step 2: Evaluate the Trained Agent
After training is complete, a `dqn_traffic_model.pth` file will be saved in the `models/` directory. You can then evaluate its performance without any random exploration:
```bash
python src/evaluate.py
```

## 5. Interpreting the Results
After training, a `training_progress.png` file is saved in the `results/` folder.
-   **Total Reward (Blue Line):** This is the most important metric. An upward trend indicates the agent is getting better at its job (minimizing wait times). The goal is to see this curve climb as high as possible (i.e., as close to zero as possible) and then plateau.
-   **Average Loss (Red Line):** This shows how "wrong" the agent's predictions are. A good loss curve will drop sharply at the beginning and then remain low and stable. The previous instability was fixed, and the current loss curve demonstrates a healthy learning process.

**Latest Training Results (200 Episodes):**

![Training Progress](results/training_progress.png)

## 6. Next Steps: Phase 3 (Federated Learning)
With a stable and effective single agent, the project is now ready for the final phase: federated learning. This involves adapting the training script to run as a **Flower client** and creating a central **Flower server** to orchestrate the learning process across multiple, independent agents.