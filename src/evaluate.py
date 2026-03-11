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
    """Parses a SUMO tripinfo XML and returns avg, max, min waiting times and completed vehicle count."""
    if not os.path.exists(filepath):
        return 0, 0, 0, 0

    tree = ET.parse(filepath)
    root = tree.getroot()

    wait_times = [float(trip.get("waitingTime"))
                  for trip in root.findall("tripinfo")]

    if not wait_times:
        return 0, 0, 0, 0

    avg_wait = np.mean(wait_times)
    max_wait = np.max(wait_times)
    min_wait = np.min(wait_times)
    completed_vehicles = len(wait_times)

    return avg_wait, max_wait, min_wait, completed_vehicles


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
            return None, None, None, None, None

        print(f"Loading model: {os.path.basename(model_path)}")
        # Action dim is 4 to choose a phase directly
        agent = TrafficLightAgent(state_dim=18, action_dim=4, device=DEVICE)
        agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.model.eval()

    total_reward = 0
    step_stats = {
        'avg_wait': [],
        'min_wait': [],
        'max_wait': [],
        'completed': [],
        'current_vehs': []
    }
    cumulative_completed = 0

    def collect_stats():
        nonlocal cumulative_completed
        ids = traci.vehicle.getIDList()
        if ids:
            waits = [traci.vehicle.getWaitingTime(v) for v in ids]
            avg_w = np.mean(waits)
            min_w = np.min(waits)
            max_w = np.max(waits)
        else:
            avg_w, min_w, max_w = 0, 0, 0
        
        cumulative_completed += traci.simulation.getArrivedNumber()
        
        step_stats['avg_wait'].append(avg_w)
        step_stats['min_wait'].append(min_w)
        step_stats['max_wait'].append(max_w)
        step_stats['completed'].append(cumulative_completed)
        step_stats['current_vehs'].append(len(ids))

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
                        # Awkward but functional way to get next green phase
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

                action = agent.select_action(state, epsilon=0.0)  # Epsilon is 0 for eval
                chosen_phase = GREEN_PHASES[action]

                if chosen_phase != current_green_phase:
                    traci.trafficlight.setPhase(
                        'J1', YELLOW_PHASES[current_green_phase])
                    for _ in range(YELLOW_TIME):
                        if step >= 1000:
                            break
                        traci.simulationStep()
                        collect_stats()
                        step += 1

                traci.trafficlight.setPhase('J1', chosen_phase)
                current_green_phase = chosen_phase
                time_in_phase = 0
                for _ in range(ACTION_COMMIT_TIME):
                    if step >= 1000:
                        break
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
        if traci.isLoaded():
            traci.close()

    avg_wait, max_wait, min_wait, completed_vehicles = get_stats_from_tripinfo(
        tripinfo_path)

    print(
        f"Evaluation complete. "
        f"Total Reward: {total_reward:.2f}, "
        f"Avg Wait: {avg_wait:.2f}s, Completed Vehicles: {completed_vehicles}"
    )

    return total_reward, avg_wait, max_wait, min_wait, completed_vehicles, step_stats

def plot_waits_comparison(waits_rl, waits_baseline, save_path):

    if waits_rl is None or waits_baseline is None:
        print("Skipping plot due to missing data.")
        return

    waits_rl = np.array(waits_rl)
    waits_baseline = np.array(waits_baseline)

    # Ensure they have the same length for plotting if needed, 
    # but plt.plot can handle different lengths.
    x_rl = np.arange(len(waits_rl))
    x_base = np.arange(len(waits_baseline))

    # Stats
    rl_mean = np.mean(waits_rl)
    rl_median = np.median(waits_rl)

    base_mean = np.mean(waits_baseline)
    base_median = np.median(waits_baseline)

    plt.figure(figsize=(12, 6))

    # RL Curve
    plt.plot(x_rl, waits_rl, label="RL", linewidth=1.0, alpha=0.7)
    plt.axhline(rl_mean, color='blue', linestyle="--", label=f"RL Mean ({rl_mean:.2f}s)")

    # Baseline Curve
    plt.plot(x_base, waits_baseline, label="Baseline", linewidth=1.0, alpha=0.7)
    plt.axhline(base_mean, color='orange', linestyle="--", label=f"Baseline Mean ({base_mean:.2f}s)")

    plt.xlabel("Simulation Step")
    plt.ylabel("Average Waiting Time Per Vehicle (s)")
    plt.title("Per-Vehicle Waiting Time Comparison (Current Vehicles)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Comparison plot saved to: {save_path}")


def plot_additional_stats(avg_wait_times, min_wait_times, max_wait_times,
                          completed_vehicles, total_vehicles, remaining_vehicles, save_path):
    plt.figure(figsize=(12, 5))

    steps = range(len(avg_wait_times))

    # Plot 1: Average Waiting Time with Min/Max Range
    plt.subplot(1, 2, 1)
    plt.plot(steps, avg_wait_times, label="Avg. Waiting Time", color="green")
    plt.fill_between(steps, min_wait_times, max_wait_times,
                     color='green', alpha=0.2, label="Min/Max Wait")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Time (s)")
    plt.title("Waiting Time Statistics over Time")
    plt.legend()
    plt.grid()

    # Plot 2: Vehicle Throughput
    plt.subplot(1, 2, 2)
    if len(total_vehicles) == 1:
        total_vehicles = [total_vehicles[0]] * len(steps)
    
    plt.plot(steps, total_vehicles,
             label="Total Vehicles Planned", color="blue", linestyle='--')
    plt.plot(steps, completed_vehicles,
             label="Completed Vehicles", color="purple")
    plt.plot(steps, remaining_vehicles,
             label="Remaining Vehicles", color="red", linestyle=':')
    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Vehicles")
    plt.title("Vehicle Throughput over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Additional stats plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained traffic control model.")
    parser.add_argument("--experiment", type=str, default="all",
                        help="Experiment name to evaluate, or 'all' to run all experiments.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a specific .pth model file (optional). Overrides default model search.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to append to the plot filename (e.g., '_custom').")
    parser.add_argument("--nogui", action="store_true",
                        help="Run SUMO without the GUI.")
    args = parser.parse_args()

    MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
    RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../results"))

    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found at: {MODELS_DIR}")
        sys.exit(1)

    experiments_to_run = []
    if args.experiment.lower() == 'all':
        experiments_to_run = [d for d in os.listdir(
            MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        if args.model:
            print("[WARNING] --model argument is ignored when --experiment is 'all'.")
    else:
        experiments_to_run = [args.experiment]
        if not os.path.isdir(os.path.join(MODELS_DIR, args.experiment)):
            print(
                f"[ERROR] Experiment '{args.experiment}' directory not found in {MODELS_DIR}.")
            sys.exit(1)

    if not experiments_to_run:
        print(f"No experiments found to evaluate.")
        sys.exit(0)

    print(
        f"Found {len(experiments_to_run)} experiments to evaluate: {experiments_to_run}\n")

    for experiment in experiments_to_run:
        print(f"--- Evaluating Experiment: {experiment} ---")

        model_path = args.model
        if not model_path:
            model_dir = os.path.join(MODELS_DIR, experiment)
            model_files = [f for f in os.listdir(
                model_dir) if f.endswith('.pth')]
            if not model_files:
                print(
                    f"  [WARNING] No model file (.pth) found for '{experiment}'. Skipping.")
                continue
            # Default to the first model found
            model_path = os.path.join(model_dir, model_files[0])

        sumo_config_base = os.path.normpath(
            os.path.join(SCRIPT_DIR, "../sumo_config"))
        route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
        tripinfo_path_rl = os.path.join(
            RESULTS_DIR, experiment, f"tripinfo_rl_{experiment}.xml")
        tripinfo_path_base = os.path.join(
            RESULTS_DIR, experiment, f"tripinfo_base_{experiment}.xml")

        os.makedirs(os.path.join(RESULTS_DIR, experiment), exist_ok=True)

        total_vehicles_in_sim = generate_routes(experiment, route_file_path)
        print(f"Generated routes for '{experiment}' with approx. {int(total_vehicles_in_sim)} vehicles.")

        stats_plot_rl_filename = f"stats_rl{args.suffix}.png"
        stats_plot_rl_path = os.path.join(
            RESULTS_DIR, experiment, stats_plot_rl_filename)

        stats_plot_baseline_filename = f"stats_baseline{args.suffix}.png"
        stats_plot_baseline_path = os.path.join(
            RESULTS_DIR, experiment, stats_plot_baseline_filename)
        
        comparison_plot_filename = f"comparison_waits{args.suffix}.png"
        comparison_plot_path = os.path.join(
            RESULTS_DIR, experiment, comparison_plot_filename)

        print("\n[1/2] Running evaluation with RL Agent...")
        reward_rl, avg_wait_rl, max_wait_rl, min_wait_rl, completed_rl, step_stats_rl = run_evaluation(
            experiment, model_path, args.nogui, use_baseline=False, tripinfo_path=tripinfo_path_rl)

        print("\n[2/2] Running evaluation with Fixed-Time Baseline...")
        reward_base, avg_wait_base, max_wait_base, min_wait_base, completed_base, step_stats_base = run_evaluation(
            experiment, model_path=None, nogui=args.nogui, use_baseline=True, tripinfo_path=tripinfo_path_base)

        print("\n--- Summary ---")

        if step_stats_rl is not None:
            remaining_rl_ts = [total_vehicles_in_sim - c for c in step_stats_rl['completed']]
            plot_additional_stats(
                step_stats_rl['avg_wait'], step_stats_rl['min_wait'], step_stats_rl['max_wait'],
                step_stats_rl['completed'], [total_vehicles_in_sim], remaining_rl_ts,
                stats_plot_rl_path
            )
            print(
                f"  RL Metrics - Reward: {reward_rl:.2f}, Avg Wait (TripInfo): {avg_wait_rl:.2f}s, "
                f"Completed: {completed_rl}/{int(total_vehicles_in_sim)}"
            )

        if step_stats_base is not None:
            remaining_base_ts = [total_vehicles_in_sim - c for c in step_stats_base['completed']]
            plot_additional_stats(
                step_stats_base['avg_wait'], step_stats_base['min_wait'], step_stats_base['max_wait'],
                step_stats_base['completed'], [total_vehicles_in_sim], remaining_base_ts,
                stats_plot_baseline_path
            )
            print(
                f"  Baseline Metrics - Reward: {reward_base:.2f}, Avg Wait (TripInfo): {avg_wait_base:.2f}s, "
                f"Completed: {completed_base}/{int(total_vehicles_in_sim)}"
            )
        
        if step_stats_rl is not None and step_stats_base is not None:
            plot_waits_comparison(
                step_stats_rl['avg_wait'], step_stats_base['avg_wait'], 
                comparison_plot_path
            )

        print("-" * (29 + len(experiment)) + "\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()