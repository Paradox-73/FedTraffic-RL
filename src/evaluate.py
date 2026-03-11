import traci
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from agent import TrafficLightAgent
from generate_routes import generate_routes
import xml.etree.ElementTree as ET

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 8-PHASE SPLIT CYCLE CONSTANTS ---
PHASE_N_GREEN = 0
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4
PHASE_S_YELLOW = 5
PHASE_W_GREEN = 6
PHASE_W_YELLOW = 7

# The Agent can only act during Green Phases
# ALLOWED_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN, PHASE_S_GREEN, PHASE_W_GREEN] # old
GREEN_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN,
                PHASE_S_GREEN, PHASE_W_GREEN]  # new

# Mapping from Green Phase to its corresponding Yellow Phase
YELLOW_PHASES = {
    PHASE_N_GREEN: PHASE_N_YELLOW,
    PHASE_E_GREEN: PHASE_E_YELLOW,
    PHASE_S_GREEN: PHASE_S_YELLOW,
    PHASE_W_GREEN: PHASE_W_YELLOW,
}

# Fixed baseline: 30s green for each of the 4 approaches
BASELINE_GREEN_TIME = 30
YELLOW_TIME = 3  # sumo default is 3s
ACTION_COMMIT_TIME = 10  # seconds, new


# --- State Function (Exact replica from control.py) ---

def get_state(last_phase_time, current_phase):
    # State Dim = 18
    q_north = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_N_in") / 20.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_S_in") / 20.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_E_in") / 20.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_W_in") / 20.0, 0, 1)

    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 100.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 100.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 100.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 100.0, 0, 1)

    is_n = 1 if current_phase == PHASE_N_GREEN else 0
    is_e = 1 if current_phase == PHASE_E_GREEN else 0
    is_s = 1 if current_phase == PHASE_S_GREEN else 0
    is_w = 1 if current_phase == PHASE_W_GREEN else 0

    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

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


def get_stats_from_tripinfo(filepath):
    """Parses a SUMO tripinfo XML and returns stats and raw wait times."""
    if not os.path.exists(filepath):
        return 0, 0, 0, 0, []

    tree = ET.parse(filepath)
    root = tree.getroot()

    wait_times = [float(trip.get("waitingTime"))
                  for trip in root.findall("tripinfo")]

    if not wait_times:
        return 0, 0, 0, 0, []

    avg_wait = np.mean(wait_times)
    max_wait = np.max(wait_times)
    min_wait = np.min(wait_times)
    completed_vehicles = len(wait_times)

    return avg_wait, max_wait, min_wait, completed_vehicles, wait_times


def run_evaluation(experiment_name, model_path, nogui, use_baseline=False, tripinfo_path=None):
    sumo_config_base = os.path.normpath(
        os.path.join(SCRIPT_DIR, "../sumo_config"))
    route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")

    sumo_bin = "sumo" if nogui else "sumo-gui"
    sumo_cmd = [
        sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end",
        "--no-warnings", "--no-step-log", "--tripinfo-output", tripinfo_path
    ]

    agent = None
    if not use_baseline:
        if not model_path or not os.path.exists(model_path):
            print(f"[ERROR] Model not found at '{model_path}'")
            return None, None, None, None, None, None

        print(f"Loading model: {os.path.basename(model_path)}")
        agent = TrafficLightAgent(state_dim=18, action_dim=4, device=DEVICE)
        agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.model.eval()

    total_reward = 0
    step_stats = {
        'total_wait': [],
        'queue_length': [],
        'active_phase': [],
        'completed': []
    }
    cumulative_completed = 0

    def collect_stats():
        nonlocal cumulative_completed
        ids = traci.vehicle.getIDList()
        
        # 1. Total waiting time at this step
        total_w = sum(traci.vehicle.getWaitingTime(v) for v in ids)
        
        # 2. Total halting vehicles (queue length)
        incoming_edges = ["edge_N_in", "edge_S_in", "edge_E_in", "edge_W_in"]
        queue_len = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)
        
        # 3. Active phase
        active_p = traci.trafficlight.getPhase('J1')
        
        cumulative_completed += traci.simulation.getArrivedNumber()
        
        step_stats['total_wait'].append(total_w)
        step_stats['queue_length'].append(queue_len)
        step_stats['active_phase'].append(active_p)
        step_stats['completed'].append(cumulative_completed)

    try:
        if traci.isLoaded():
            traci.close()
        traci.start(sumo_cmd)

        step = 0
        time_in_phase = 0
        current_green_phase = traci.trafficlight.getPhase('J1')

        while step < 1000:
            if use_baseline:
                # --- Baseline Fixed-Time Logic ---
                current_phase_id = traci.trafficlight.getPhase('J1')
                if current_phase_id in GREEN_PHASES:
                    if time_in_phase >= BASELINE_GREEN_TIME:
                        traci.trafficlight.setPhase(
                            'J1', YELLOW_PHASES[current_phase_id])
                        time_in_phase = 0
                else:  # Yellow Phase
                    if time_in_phase >= YELLOW_TIME:
                        next_green_phase_idx = (
                            GREEN_PHASES.index(current_green_phase) + 1) % len(GREEN_PHASES)
                        current_green_phase = GREEN_PHASES[next_green_phase_idx]
                        traci.trafficlight.setPhase(
                            'J1', current_green_phase)
                        time_in_phase = 0
                traci.simulationStep()
                collect_stats()
                step += 1
                time_in_phase += 1
            else:
                # --- RL Action-Commitment Logic ---
                state = get_state(time_in_phase, current_green_phase)
                wait_at_start_of_chunk = sum(
                    traci.vehicle.getWaitingTime(v_id) for v_id in traci.vehicle.getIDList())

                action = agent.select_action(state, epsilon=0.0)
                chosen_phase = GREEN_PHASES[action]

                if chosen_phase != current_green_phase:
                    traci.trafficlight.setPhase(
                        'J1', YELLOW_PHASES[current_green_phase])
                    for _ in range(YELLOW_TIME):
                        if step >= 1000: break
                        traci.simulationStep()
                        collect_stats()
                        step += 1

                traci.trafficlight.setPhase('J1', chosen_phase)
                current_green_phase = chosen_phase
                time_in_phase = 0
                for _ in range(ACTION_COMMIT_TIME):
                    if step >= 1000: break
                    traci.simulationStep()
                    collect_stats()
                    step += 1
                    time_in_phase += 1
                
                wait_at_end_of_chunk = sum(
                    traci.vehicle.getWaitingTime(v_id) for v_id in traci.vehicle.getIDList())
                chunk_reward = (wait_at_start_of_chunk - wait_at_end_of_chunk) / 100.0
                total_reward += chunk_reward

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None
    finally:
        if traci.isLoaded(): traci.close()

    avg_wait, max_wait, min_wait, completed_vehicles, wait_times = get_stats_from_tripinfo(tripinfo_path)

    print(f"Evaluation complete. Reward: {total_reward:.2f}, Avg Wait: {avg_wait:.2f}s, Completed: {completed_vehicles}")
    return total_reward, avg_wait, max_wait, min_wait, completed_vehicles, step_stats, wait_times


def plot_cumulative_waiting_time(waits_rl, waits_base, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(waits_rl), label="RL Agent", color="blue", linewidth=2)
    plt.plot(np.cumsum(waits_base), label="Baseline", color="orange", linewidth=2)
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Waiting Time (s)")
    plt.title("Cumulative Waiting Time Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_queue_length(queues_rl, queues_base, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(queues_rl, label="RL Agent", color="blue", alpha=0.4)
    plt.plot(queues_base, label="Baseline", color="orange", alpha=0.4)
    
    # Add smoothed lines
    window = 20
    if len(queues_rl) > window:
        rl_ma = np.convolve(queues_rl, np.ones(window)/window, mode='valid')
        base_ma = np.convolve(queues_base, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(queues_rl)), rl_ma, color="darkblue", linewidth=2, label="RL (MA)")
        plt.plot(range(window-1, len(queues_base)), base_ma, color="darkorange", linewidth=2, label="Baseline (MA)")
    
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length (Vehicles)")
    plt.title("Queue Length vs. Simulation Step (Halted Vehicles)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_waiting_time_distribution(trips_rl, trips_base, save_path):
    plt.figure(figsize=(8, 6))
    plt.boxplot([trips_rl, trips_base], labels=["RL Agent", "Baseline"], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    plt.ylabel("Waiting Time (s)")
    plt.title("Waiting Time Distribution (Completed Vehicles)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_phase_timeline(phases_rl, phases_base, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    phase_labels = {0: 'N Green', 1: 'N Yellow', 2: 'E Green', 3: 'E Yellow',
                    4: 'S Green', 5: 'S Yellow', 6: 'W Green', 7: 'W Yellow'}
    
    def plot_timeline(ax, phases, title):
        phases = np.array(phases)
        changes = np.where(np.diff(phases) != 0)[0] + 1
        starts = np.insert(changes, 0, 0)
        ends = np.append(changes, len(phases))
        colors = {0: 'lime', 1: 'gold', 2: 'forestgreen', 3: 'khaki', 
                  4: 'green', 5: 'darkkhaki', 6: 'darkgreen', 7: 'olive'}
        for start, end in zip(starts, ends):
            phase = phases[start]
            ax.broken_barh([(start, end-start)], (phase, 0.8), facecolors=colors.get(phase, 'gray'))
        ax.set_yticks([i + 0.4 for i in range(8)])
        ax.set_yticklabels([phase_labels[i] for i in range(8)])
        ax.set_title(title)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plot_timeline(ax1, phases_rl, "RL Agent Phase Timeline")
    plot_timeline(ax2, phases_base, "Baseline Phase Timeline")
    plt.xlabel("Simulation Step")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained traffic control model.")
    parser.add_argument("--experiment", type=str, default="all", help="Experiment name or 'all'")
    parser.add_argument("--model", type=str, default=None, help="Path to a specific .pth model file")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for plot filenames")
    parser.add_argument("--nogui", action="store_true", help="Run SUMO without the GUI.")
    args = parser.parse_args()

    MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
    RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../results"))

    experiments_to_run = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))] if args.experiment.lower() == 'all' else [args.experiment]

    for experiment in experiments_to_run:
        print(f"--- Evaluating Experiment: {experiment} ---")
        model_dir = os.path.join(MODELS_DIR, experiment)
        model_path = args.model or os.path.join(model_dir, [f for f in os.listdir(model_dir) if f.endswith('.pth')][0])
        
        sumo_config_base = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config"))
        tripinfo_path_rl = os.path.join(RESULTS_DIR, experiment, f"tripinfo_rl_{experiment}.xml")
        tripinfo_path_base = os.path.join(RESULTS_DIR, experiment, f"tripinfo_base_{experiment}.xml")
        os.makedirs(os.path.join(RESULTS_DIR, experiment), exist_ok=True)

        total_vehicles_in_sim = generate_routes(experiment, os.path.join(sumo_config_base, "hello.rou.xml"))

        print("\n[1/2] Running RL Agent...")
        res_rl = run_evaluation(experiment, model_path, args.nogui, False, tripinfo_path_rl)
        
        print("\n[2/2] Running Baseline...")
        res_base = run_evaluation(experiment, None, args.nogui, True, tripinfo_path_base)

        if res_rl[5] and res_base[5]:
            # Generate the 4 new plots
            plot_cumulative_waiting_time(res_rl[5]['total_wait'], res_base[5]['total_wait'], 
                                         os.path.join(RESULTS_DIR, experiment, f"eval_cumulative_wait{args.suffix}.png"))
            plot_queue_length(res_rl[5]['queue_length'], res_base[5]['queue_length'], 
                              os.path.join(RESULTS_DIR, experiment, f"eval_queue_length{args.suffix}.png"))
            plot_waiting_time_distribution(res_rl[6], res_base[6], 
                                          os.path.join(RESULTS_DIR, experiment, f"eval_wait_dist{args.suffix}.png"))
            plot_phase_timeline(res_rl[5]['active_phase'], res_base[5]['active_phase'], 
                                os.path.join(RESULTS_DIR, experiment, f"eval_phase_timeline{args.suffix}.png"))
            print(f"Generated evaluation plots in {os.path.join(RESULTS_DIR, experiment)}")

if __name__ == "__main__":
    main()