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
ALLOWED_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN, PHASE_S_GREEN, PHASE_W_GREEN]

# Fixed baseline: 30s green for each of the 4 approaches
BASELINE_GREEN_TIME = 30
YELLOW_TIME = 3  # sumo default is 3s

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

    # generate_routes is called outside this function to ensure both RL and Baseline use the same traffic.
    # total_vehicles_in_sim is now passed as an argument.

    sumo_bin = "sumo" if nogui else "sumo-gui"
    sumo_cmd = [
        sumo_bin,
        "-c", sumo_cfg_path,
        "--start",
        "--quit-on-end",
        "--no-warnings",
        "--no-step-log",
        "--tripinfo-output", tripinfo_path # Added tripinfo output
    ]

    agent = None
    if not use_baseline:
        if not model_path or not os.path.exists(model_path):
            print(f"[ERROR] Model not found at '{model_path}'")
            # Return None for all expected values
            return None, None, None, None, None, None

        print(f"Loading model: {os.path.basename(model_path)}")
        agent = TrafficLightAgent(state_dim=18, action_dim=2, device=DEVICE)
        agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.model.eval()

    waits_per_step = []
    total_reward = 0

    try:
        if traci.isLoaded():
            traci.close()

        traci.start(sumo_cmd)

        step = 0
        last_phase_time = 0
        previous_avg_wait = 0.0

        while step < 1000:
            current_phase_id = traci.trafficlight.getPhase('J1')

            # --- Control Logic ---
            if use_baseline:
                if current_phase_id in ALLOWED_PHASES:
                    if last_phase_time >= BASELINE_GREEN_TIME:
                        traci.trafficlight.setPhase('J1', current_phase_id + 1)
                        last_phase_time = 0
                else:
                    if last_phase_time >= YELLOW_TIME:
                        next_green_phase = (current_phase_id + 1) % 8
                        traci.trafficlight.setPhase('J1', next_green_phase)
                        last_phase_time = 0
            else:
                if current_phase_id in ALLOWED_PHASES:
                    state = get_state(last_phase_time, current_phase_id)
                    action = agent.select_action(state, epsilon=0.0)
                    if action == 1:
                        traci.trafficlight.setPhase('J1', current_phase_id + 1)
                        last_phase_time = 0

            traci.simulationStep()
            step += 1
            last_phase_time += 1

            # --- Per-vehicle waiting time ---
            vehicle_ids = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_ids)

            if num_vehicles > 0:
                total_wait = sum(
                    traci.vehicle.getWaitingTime(v) for v in vehicle_ids
                )
                avg_wait = total_wait / num_vehicles
            else:
                avg_wait = 0.0

            waits_per_step.append(avg_wait)

            # Reward = reduction in per-vehicle wait
            reward = previous_avg_wait - avg_wait
            previous_avg_wait = avg_wait
            total_reward += reward

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        return None, None, None, None, None, None

    finally:
        if traci.isLoaded():
            traci.close()

    # Get stats from tripinfo after closing traci
    avg_wait, max_wait, min_wait, completed_vehicles = get_stats_from_tripinfo(
        tripinfo_path)

    print(
        f"Evaluation complete. Total Reward: {total_reward:.2f}, "
        f"Avg Wait: {avg_wait:.2f}s, Completed Vehicles: {completed_vehicles}"
    )

    return waits_per_step, total_reward, avg_wait, max_wait, min_wait, completed_vehicles

def plot_waits_comparison(waits_rl, waits_baseline, save_path):

    if waits_rl is None or waits_baseline is None:
        print("Skipping plot due to missing data.")
        return

    waits_rl = np.array(waits_rl)
    waits_baseline = np.array(waits_baseline)

    x = np.arange(len(waits_rl))

    # Stats
    rl_mean = np.mean(waits_rl)
    rl_median = np.median(waits_rl)

    base_mean = np.mean(waits_baseline)
    base_median = np.median(waits_baseline)

    plt.figure(figsize=(12, 6))

    # RL Curve
    plt.plot(x, waits_rl, label="RL", linewidth=1.5)
    plt.axhline(rl_mean, linestyle="--", label="RL Mean")
    plt.axhline(rl_median, linestyle=":", label="RL Median")

    # Baseline Curve
    plt.plot(x, waits_baseline, label="Baseline", linewidth=1.5)
    plt.axhline(base_mean, linestyle="--", label="Baseline Mean")
    plt.axhline(base_median, linestyle=":", label="Baseline Median")

    plt.xlabel("Simulation Step")
    plt.ylabel("Average Waiting Time Per Vehicle (s)")
    plt.title("Per-Vehicle Waiting Time Comparison")
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

    # Plot 1: Average Waiting Time with Min/Max Range
    plt.subplot(1, 2, 1)
    episodes = range(len(avg_wait_times))
    plt.plot(episodes, avg_wait_times, label="Avg. Waiting Time", color="green")
    plt.fill_between(episodes, min_wait_times, max_wait_times,
                     color='green', alpha=0.2, label="Min/Max Wait")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time (s)")
    plt.title("Waiting Time Statistics per Episode")
    plt.legend()
    plt.grid()

    # Plot 2: Vehicle Throughput
    plt.subplot(1, 2, 2)
    plt.plot(episodes, total_vehicles,
             label="Total Vehicles in Sim", color="blue", linestyle='--')
    plt.plot(episodes, completed_vehicles,
             label="Completed Vehicles", color="purple")
    plt.plot(episodes, remaining_vehicles,
             label="Remaining Vehicles", color="red", linestyle=':')
    plt.xlabel("Episode")
    plt.ylabel("Number of Vehicles")
    plt.title("Vehicle Throughput per Episode")
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
            model_path = os.path.join(model_dir, model_files[0])

        sumo_config_base = os.path.normpath(
            os.path.join(SCRIPT_DIR, "../sumo_config"))
        route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
        tripinfo_path = os.path.join(
            RESULTS_DIR, experiment, f"tripinfo_{experiment}.xml")

        # Ensure results directory exists for this experiment
        os.makedirs(os.path.join(RESULTS_DIR, experiment), exist_ok=True)

        # Generate routes once for the experiment, both RL and baseline will use this traffic
        total_vehicles_in_sim = generate_routes(experiment, route_file_path)

        plot_comparison_filename = f"comparison_plot{args.suffix}.png"
        result_plot_path = os.path.join(
            RESULTS_DIR, experiment, plot_comparison_filename)

        stats_plot_rl_filename = f"stats_rl{args.suffix}.png"
        stats_plot_rl_path = os.path.join(
            RESULTS_DIR, experiment, stats_plot_rl_filename)

        stats_plot_baseline_filename = f"stats_baseline{args.suffix}.png"
        stats_plot_baseline_path = os.path.join(
            RESULTS_DIR, experiment, stats_plot_baseline_filename)

        print("\n[1/2] Running evaluation with RL Agent...")
        waits_rl, reward_rl, avg_wait_rl, max_wait_rl, min_wait_rl, completed_rl = run_evaluation(
            experiment, model_path, args.nogui, use_baseline=False, tripinfo_path=tripinfo_path)

        print("\n[2/2] Running evaluation with Fixed-Time Baseline...")
        waits_base, reward_base, avg_wait_base, max_wait_base, min_wait_base, completed_base = run_evaluation(
            experiment, model_path=None, nogui=args.nogui, use_baseline=True, tripinfo_path=tripinfo_path)

        print("\n--- Summary ---")
        plot_waits_comparison(waits_rl, waits_base, result_plot_path)

        if all(x is not None for x in [avg_wait_rl, min_wait_rl, max_wait_rl, completed_rl]):
            remaining_rl = total_vehicles_in_sim - completed_rl
            plot_additional_stats(
                [avg_wait_rl], [min_wait_rl], [max_wait_rl],
                [completed_rl], [total_vehicles_in_sim], [remaining_rl],
                stats_plot_rl_path
            )
            print(f"  RL Metrics - Avg Wait: {avg_wait_rl:.2f}s, Completed: {completed_rl}/{total_vehicles_in_sim}, Remaining: {remaining_rl}")

        if all(x is not None for x in [avg_wait_base, min_wait_base, max_wait_base, completed_base]):
            remaining_base = total_vehicles_in_sim - completed_base
            plot_additional_stats(
                [avg_wait_base], [min_wait_base], [max_wait_base],
                [completed_base], [total_vehicles_in_sim], [remaining_base],
                stats_plot_baseline_path
            )
            print(f"  Baseline Metrics - Avg Wait: {avg_wait_base:.2f}s, Completed: {completed_base}/{total_vehicles_in_sim}, Remaining: {remaining_base}")

        if reward_rl is not None and reward_base is not None:
            print(f"  Total Reward (RL): {reward_rl:.2f}")
            print(f"  Total Reward (Baseline): {reward_base:.2f}")

        print("-" * (29 + len(experiment)) + "\n")

if __name__ == "__main__":
    main()