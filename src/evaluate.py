import traci
import sys
import os
import numpy as np
import torch
from agent import TrafficLightAgent
import argparse

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Traffic Light Phases for J1
PHASE_NS_GREEN_ID = 0
PHASE_NS_YELLOW_ID = 1
PHASE_EW_GREEN_ID = 2
PHASE_EW_YELLOW_ID = 3

# Fixed baseline: 90 s green each phase, 3 s yellow
BASELINE_GREEN = 90
BASELINE_YELLOW = 3

def get_state(last_phase_time, current_phase):
    """
    Exact replica of the get_state function from control.py
    State Dim = 15
    """
    # Standard Queue and Waiting times
    q_north = np.clip(traci.edge.getLastStepHaltingNumber("edge_N_in") / 20.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber("edge_S_in") / 20.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber("edge_E_in") / 20.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber("edge_W_in") / 20.0, 0, 1)
    
    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 100.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 100.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 100.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 100.0, 0, 1)
    
    # Phase information
    is_ns_green = 1 if current_phase == PHASE_NS_GREEN_ID else 0
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)
    
    # Pressure (Incoming - Outgoing)
    # Note: If edges aren't exactly 20 capacity, this normalization is approximate, 
    # but it must match training exactly.
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber("edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 20.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber("edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 20.0, -1, 1)
    pressure_east = np.clip((traci.edge.getLastStepVehicleNumber("edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 20.0, -1, 1)
    pressure_west = np.clip((traci.edge.getLastStepVehicleNumber("edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 20.0, -1, 1)

    return np.array([q_north, q_south, q_east, q_west, 
                     wait_north, wait_south, wait_east, wait_west, 
                     is_ns_green, norm_last_phase_time, 
                     pressure_north, pressure_south, pressure_east, pressure_west, 
                     1.0])

def run_evaluation(experiment_name, nogui, compare_baseline=False):
    # Paths
    sumo_config_base = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config"))
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")
    
    # We must load the route file specific to the experiment to ensure the evaluation
    # scenario matches the training scenario name (even if the file is physically hello.rou.xml,
    # we assume the user might have different route generations)
    # Ideally, generate the specific route first or assume hello.rou.xml is already set up.
    
    model_path = os.path.normpath(os.path.join(SCRIPT_DIR, f"../models/{experiment_name}/dqn_traffic_model.pth"))
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")

    # Initialize Agent with State Dim 15
    agent = TrafficLightAgent(state_dim=15, action_dim=2, device=DEVICE)
    try:
        agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.model.eval()
    except RuntimeError as e:
        print(f"Error loading model weights. Did you change state_dim? \nDetails: {e}")
        return

    sumo_bin = "sumo" if nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end", "--no-warnings"]
    
    try:
        traci.start(sumo_cmd)
        step = 0
        last_phase_time = 0
        total_reward = 0
        previous_total_wait = 0
        
        # Initial State
        current_phase = traci.trafficlight.getPhase('J1')
        state = get_state(last_phase_time, current_phase)

        print("Starting Evaluation...")
        
        while step < 1000:
            current_phase_id = traci.trafficlight.getPhase('J1')

            if compare_baseline:
                # Hard-coded controller: 90s green then 3s yellow, cycling NS -> EW
                phase_time = last_phase_time
                if current_phase_id in (PHASE_NS_GREEN_ID, PHASE_EW_GREEN_ID):
                    if phase_time >= BASELINE_GREEN:
                        traci.trafficlight.setPhase('J1', current_phase_id + 1)
                        last_phase_time = 0
                else:  # yellow
                    if phase_time >= BASELINE_YELLOW:
                        # move to the opposite green
                        traci.trafficlight.setPhase('J1', (current_phase_id + 1) % 4)
                        last_phase_time = 0
            else:
                # RL agent acts on green phases
                if current_phase_id == PHASE_NS_GREEN_ID or current_phase_id == PHASE_EW_GREEN_ID:
                    action = agent.select_action(state, epsilon=0.0) 
                    
                    if action == 1:
                        traci.trafficlight.setPhase('J1', current_phase_id + 1)
                        last_phase_time = 0
                else:
                    # During yellow, action is irrelevant, just wait
                    pass

            traci.simulationStep()
            step += 1
            last_phase_time += 1
            
            # Update state for next step
            next_phase = traci.trafficlight.getPhase('J1')
            state = get_state(last_phase_time, next_phase)
            
            # Calculate Reward (just for reporting)
            vehicle_ids = traci.vehicle.getIDList()
            current_total_wait = sum(traci.vehicle.getWaitingTime(v_id) for v_id in vehicle_ids)
            reward = (previous_total_wait - current_total_wait) / 100.0
            previous_total_wait = current_total_wait
            
            total_reward += reward

        print(f"Evaluation Complete. Total Accumulated Reward: {total_reward:.2f}")
    
    except Exception as e:
        print(f"Simulation Error: {e}")
    finally:
        if traci.isLoaded(): traci.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="stage1_baseline", help="Experiment name folder to load model from")
    parser.add_argument("--nogui", action="store_true", help="Run without GUI")
    parser.add_argument("--baseline", action="store_true", help="Run fixed 90s-per-approach controller instead of RL model")
    args = parser.parse_args()
    
    run_evaluation(args.experiment, args.nogui, compare_baseline=args.baseline)
