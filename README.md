# Federated Reinforcement Learning for Adaptive Traffic Signal Control

This project implements a decentralized traffic control system where traffic lights learn to optimize flow using Deep Reinforcement Learning (DQN) and collaborate via Federated Learning (Flower).

## 📂 Project Structure
- `hello.sumocfg`: Main configuration file for SUMO.
- `hello.net.xml`: The road network (Single 4-way intersection).
- `hello.rou.xml`: Traffic demand (Heterogeneous: Cars, Buses, Motorcycles).
- `control.py`: Python controller that interfaces with SUMO (State/Reward extraction).

## 🚀 Getting Started (Member 1 - Simulator)

### Prerequisites
1. Install **SUMO** (Simulation of Urban MObility).
2. Install Python dependencies:
   ```bash
   pip install traci torch flwr

```

### Running the Simulation

To launch the traffic simulation with the rule-based controller:

```bash
python control.py

```

This will:

* Open the SUMO-GUI.
* Spawn heterogeneous traffic (Red Buses, Blue Motos, Yellow Cars).
* Output the **State** (Queue lengths) and **Reward** (Wait times) to the terminal.

## 🧠 System Architecture (For Team Members)

### State Space (Input to AI)

A vector of size `4` representing halting vehicles on incoming lanes:
`[Queue_North, Queue_South, Queue_East, Queue_West]`

### Action Space (Output of AI)

* `0`: Keep current phase.
* `1`: Switch to the next phase.

### Reward Function


*Objective: Maximize Reward (i.e., minimize waiting time).*

```

## 🛠 Development Notes
**Regenerating the Map:**
If you modify `hello.nod.xml` or `hello.edg.xml`, you must recompile the network file:
```bash
netconvert --node-files hello.nod.xml --edge-files hello.edg.xml -o hello.net.xml

```

## 📅 Project Roadmap
- [x] **Phase 1: Environment Setup** (SUMO Simulation, Heterogeneous Traffic, State/Reward API)
- [ ] **Phase 2: RL Agent** (Implement `TrafficLightDQN` in PyTorch) [Member 2]
- [ ] **Phase 3: Federated Learning** (Setup Flower Server & Client wrapper) [Member 3]
- [ ] **Phase 4: Scaling** (Expand to 4x4 Grid)

```

## ❓ Troubleshooting
**Error: `SUMO_HOME` is not set properly**
- Ensure you have added the SUMO bin folder to your system PATH.
- On Windows, set a System Variable `SUMO_HOME` pointing to your install directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`).

