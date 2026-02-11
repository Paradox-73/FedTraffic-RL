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

# --- NEW 8-PHASE SPLIT CYCLE CONSTANTS ---
# Phases match the TLS program in hello.net.xml:
#   0: North green (right/straight/left)   | 1: North yellow
#   2: East  green                         | 3: East  yellow
#   4: South green                         | 5: South yellow
#   6: West  green                         | 7: West  yellow
PHASE_N_GREEN = 0
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4
PHASE_S_YELLOW = 5
PHASE_W_GREEN = 6
PHASE_W_YELLOW = 7

# The agent is only allowed to act during green phases.
ALLOWED_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN, PHASE_S_GREEN, PHASE_W_GREEN]

def get_state(last_phase_time, current_phase):
    # State Dim = 18
    # Breakdown:
    #   4 (Queues)
    # + 4 (Waits)
    # + 4 (Phase One-Hot: [N, E, S, W]) <--- CHANGED
    # + 1 (Time)
    # + 4 (Pressure)
    # + 1 (Bias)
    # = 18 Inputs

    # 1. QUEUES
    q_north = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_N_in") / 20.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_S_in") / 20.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_E_in") / 20.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_W_in") / 20.0, 0, 1)

    # 2. WAITS
    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 100.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 100.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 100.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 100.0, 0, 1)

    # 3. PHASE CONTEXT (One-Hot Encoding)
    # Which direction is currently green?
    is_n = 1 if current_phase == PHASE_N_GREEN else 0
    is_e = 1 if current_phase == PHASE_E_GREEN else 0
    is_s = 1 if current_phase == PHASE_S_GREEN else 0
    is_w = 1 if current_phase == PHASE_W_GREEN else 0

    # How long has it been green?
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

    # 4. PRESSURE
    # Simple pressure: Incoming - Outgoing (Simplification for single lane)
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 20.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 20.0, -1, 1)
    pressure_east = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 20.0, -1, 1)
    pressure_west = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 20.0, -1, 1)

    return np.array([q_north, q_south, q_east, q_west,
                     wait_north, wait_south, wait_east, wait_west,
                     is_n, is_e, is_s, is_w,
                     norm_last_phase_time,
                     pressure_north, pressure_south, pressure_east, pressure_west,
                     1.0])

def run_evaluation(experiment_name, nogui):
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

    # Same state/action space that was used during training (includes turning traffic)
    agent = TrafficLightAgent(state_dim=18, action_dim=2, device=DEVICE)
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

            # Agent only acts on green phases (turns included in TLS definition)
            if current_phase_id in ALLOWED_PHASES:
                # Epsilon 0.0 means purely greedy (exploitation)
                action = agent.select_action(state, epsilon=0.0)

                if action == 1:
                    # Move to the yellow phase for the same approach
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
    args = parser.parse_args()

    run_evaluation(args.experiment, args.nogui)