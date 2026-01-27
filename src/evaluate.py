import traci
import sys
import os
import numpy as np
import torch
from agent import TrafficLightAgent
import time

# --- Configuration ---
# Build absolute paths based on the script's location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config/hello.sumocfg"))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
MODEL_PATH = os.path.join(MODELS_DIR, "dqn_traffic_model.pth")

SUMO_CMD = ["sumo-gui", "-c", SUMO_CFG_PATH, "--start", "--quit-on-end"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(last_phase_time, current_phase):
    """
    Returns the 12-dimensional state vector for the agent.
    """
    q_north = traci.edge.getLastStepHaltingNumber("edge_N_in")
    q_south = traci.edge.getLastStepHaltingNumber("edge_S_in")
    q_east = traci.edge.getLastStepHaltingNumber("edge_E_in")
    q_west = traci.edge.getLastStepHaltingNumber("edge_W_in")

    wait_north = traci.edge.getWaitingTime("edge_N_in")
    wait_south = traci.edge.getWaitingTime("edge_S_in")
    wait_east = traci.edge.getWaitingTime("edge_E_in")
    wait_west = traci.edge.getWaitingTime("edge_W_in")
    
    is_ns_green = 1 if current_phase in [0, 1] else 0
    is_yellow = 1 if current_phase in [1, 3] else 0

    return np.array([
        q_north, q_south, q_east, q_west,
        wait_north, wait_south, wait_east, wait_west,
        is_ns_green,
        last_phase_time,
        is_yellow,
        1.0
    ])

def run_evaluation():
    """
    Runs the simulation with the trained agent in evaluation mode.
    """
    print(f"🧠 Loading trained model from {MODEL_PATH}...")
    agent = TrafficLightAgent(state_dim=12, action_dim=2, device=DEVICE)
    try:
        agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}", file=sys.stderr)
        print("Please run 'python src/control.py' to train and save the model first.", file=sys.stderr)
        return
        
    agent.model.eval() # Set the model to evaluation mode

    traci.start(SUMO_CMD)
    print("🚀 Simulator Started for Evaluation.")

    step = 0
    last_phase_time = 0
    total_reward = 0
    
    current_phase = traci.trafficlight.getPhase('J1')
    state = get_state(last_phase_time, current_phase)

    while step < 5000: # Run for a long single episode
        traci.simulationStep()
        
        # Use epsilon=0 for pure exploitation of learned policy
        action = agent.select_action(state, epsilon=0)

        if action == 1:
            traci.trafficlight.setPhase('J1', (current_phase + 1) % 4)
            for _ in range(5):
                if step < 5000:
                    traci.simulationStep()
                    step += 1
            traci.trafficlight.setPhase('J1', (current_phase + 2) % 4)
            last_phase_time = 0
        
        current_phase = traci.trafficlight.getPhase('J1')
        reward = -sum(traci.vehicle.getWaitingTime(v_id) for v_id in traci.vehicle.getIDList())
        state = get_state(last_phase_time, current_phase)
        
        total_reward += reward
        step += 1
        last_phase_time += 1
        
        if step % 100 == 0:
            print(f"Step {step}/5000 | Current Total Reward: {total_reward:.2f}")

    print(f"\n✅ Evaluation finished. Final Total Reward: {total_reward:.2f}")
    traci.close()

if __name__ == "__main__":
    try:
        run_evaluation()
    except traci.TraCIException as e:
        print(f"\n❌ A SUMO error occurred: {e}", file=sys.stderr)
        print("Please ensure SUMO is installed and SUMO_HOME is set correctly.", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if traci.isLoaded():
            traci.close()