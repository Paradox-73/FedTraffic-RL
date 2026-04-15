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
import config
from collections import deque

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = config.DEVICE

# In get_state or collect_stats
def get_junction_queue(edge_id):
    # Count all vehicles halted on the edge
    return traci.edge.getLastStepHaltingNumber(edge_id)

# --- State Function ---

def get_state(last_phase_time, current_phase, time_since_green):
    # State Dim = 22
    # 1. QUEUES 
    q_north = np.clip(traci.edge.getLastStepHaltingNumber("edge_N_in") / 60.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber("edge_S_in") / 60.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber("edge_E_in") / 60.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber("edge_W_in") / 60.0, 0, 1)

    # 2. WAITS
    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 2000.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 2000.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 2000.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 2000.0, 0, 1)

    # 3. PHASE CONTEXT
    is_n = 1 if current_phase == config.PHASE_N_GREEN else 0
    is_e = 1 if current_phase == config.PHASE_E_GREEN else 0
    is_s = 1 if current_phase == config.PHASE_S_GREEN else 0
    is_w = 1 if current_phase == config.PHASE_W_GREEN else 0

    # 4. TIME IN PHASE
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

    # 5. PRESSURE
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber("edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 60.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber("edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 60.0, -1, 1)
    pressure_east = np.clip((traci.edge.getLastStepVehicleNumber("edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 60.0, -1, 1)
    pressure_west = np.clip((traci.edge.getLastStepVehicleNumber("edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 60.0, -1, 1)

    # 6. STARVATION (The missing 4 parameters)
    starve_n = np.clip(time_since_green[config.PHASE_N_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_e = np.clip(time_since_green[config.PHASE_E_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_s = np.clip(time_since_green[config.PHASE_S_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_w = np.clip(time_since_green[config.PHASE_W_GREEN] / config.STARVATION_NORM, 0, 1)

    return np.array([q_north, q_south, q_east, q_west,
                     wait_north, wait_south, wait_east, wait_west,
                     is_n, is_e, is_s, is_w,
                     norm_last_phase_time,
                     pressure_north, pressure_south, pressure_east, pressure_west,
                     1.0,
                     starve_n, starve_e, starve_s, starve_w], dtype=np.float32)


def get_stats_from_tripinfo(filepath):
    if not os.path.exists(filepath):
        return 0, 0, 0, 0, []
    tree = ET.parse(filepath)
    root = tree.getroot()
    wait_times = [float(trip.get("waitingTime"))
                  for trip in root.findall("tripinfo")]
    if not wait_times:
        return 0, 0, 0, 0, []
    return np.mean(wait_times), np.max(wait_times), np.min(wait_times), len(wait_times), wait_times


def run_evaluation(experiment_name, model_path, nogui, use_baseline=False, tripinfo_path=None):
    sumo_config_base = os.path.normpath(
        os.path.join(SCRIPT_DIR, "../sumo_config"))
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")
    sumo_bin = "sumo" if nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path,
                # "--delay", "100",
                "--start", "--quit-on-end",
                "--no-warnings", "--no-step-log", "--tripinfo-output", tripinfo_path]

    agent = None
    if not use_baseline:
        agent = TrafficLightAgent(state_dim=22, action_dim=4)
        agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.model.eval()

    cumulative_reward = 0
    cumulative_completed = 0
    step_stats = {'total_wait': [], 'queue_length': [], 'active_phase': [],
                  'completed': [], 'cumulative_reward': [], 'step_reward': [], 'flow_rate': []}

    # Initialize tracking for flow rate
    prev_outgoing_vehicles = {edge: set() for edge in config.OUTGOING_EDGES}
    flow_window = deque(maxlen=10)  # For smoothing flow rate over last 3 steps

    def collect_stats():
        nonlocal cumulative_completed, cumulative_reward

        # --- REWARD LOGIC (Flow Rate vs Waiting Cars vs Starvation) ---
        flow_count = 0
        for edge in config.OUTGOING_EDGES:
            current_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
            new_vehicles = current_vehicles - prev_outgoing_vehicles[edge]
            flow_count += len(new_vehicles)
            prev_outgoing_vehicles[edge] = current_vehicles

        # Flow rate for this step (time = 1)
        flow_rate = flow_count / 1.0
        flow_window.append(flow_rate)
        windowed_flow_rate = sum(flow_window) / len(flow_window)

        halting_cars = sum(get_junction_queue(edge)
                           for edge in config.INCOMING_EDGES)

        # Calculate starvation penalty to match control.py
        starvation_penalty = sum(
            time_since_green[p]
            for p in config.GREEN_PHASES
            if p != current_green_phase
        ) / config.STARVATION_NORM

        step_reward = (config.W1 * windowed_flow_rate) - (config.W2 * halting_cars) - (config.W3 * starvation_penalty)
        cumulative_reward += step_reward
        cumulative_completed += flow_count

        step_stats['total_wait'].append(
            sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()))
        step_stats['queue_length'].append(halting_cars)
        step_stats['active_phase'].append(traci.trafficlight.getPhase('J1'))
        step_stats['completed'].append(cumulative_completed)
        step_stats['cumulative_reward'].append(cumulative_reward)
        step_stats['step_reward'].append(step_reward)
        step_stats['flow_rate'].append(windowed_flow_rate)

    try:
        if traci.isLoaded():
            traci.close()
        traci.start(sumo_cmd)
        # Allow SUMO to initialize properly
        step = 0
        time_in_phase = 0
        current_green_phase = traci.trafficlight.getPhase('J1')

        # Initialize time_since_green at the start of the evaluation episode
        time_since_green = {p: 0 for p in config.GREEN_PHASES}

        print("Baseline Green Phase Time: {}s | Yellow Phase Time: {}s".format(
            config.BASELINE_GREEN_TIME, config.YELLOW_TIME))

        while step < config.SIMULATION_TIME:

            # Update time_since_green every step
            for p in config.GREEN_PHASES:
                if p == current_green_phase:
                    time_since_green[p] = 0
                else:
                    time_since_green[p] += 1
            
            # CHANING COLOR BASED ON SPEED
            for veh_id in traci.vehicle.getIDList():
                # Speed < 0.1 m/s is typically considered halted in SUMO
                if traci.vehicle.getSpeed(veh_id) < 0.1:
                    # Set to Red (R, G, B, A)
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255)) 
                else:
                    # Reset to your default Yellow defined in the vType
                    traci.vehicle.setColor(veh_id, (255, 255, 0, 255))

            if use_baseline:
                # Baseline logic (Fixed-Time)
                current_phase_id = traci.trafficlight.getPhase('J1')
                if current_phase_id in config.GREEN_PHASES:
                    if time_in_phase >= config.BASELINE_GREEN_TIME:
                        traci.trafficlight.setPhase(
                            'J1', config.YELLOW_PHASES[current_phase_id])
                        time_in_phase = 0
                else:
                    if time_in_phase >= config.YELLOW_TIME:
                        next_idx = (config.GREEN_PHASES.index(
                            current_green_phase) + 1) % 4
                        current_green_phase = config.GREEN_PHASES[next_idx]
                        traci.trafficlight.setPhase('J1', current_green_phase)
                        time_in_phase = 0
                traci.simulationStep()
                collect_stats()
                step += 1
                time_in_phase += 1
            else:
                # Call get_state with the new signature
                state = get_state(time_in_phase, current_green_phase, time_since_green)
                action = agent.select_action(state, epsilon=0.0)
                chosen_phase = config.GREEN_PHASES[action]

                if chosen_phase != current_green_phase:
                    traci.trafficlight.setPhase(
                        'J1', config.YELLOW_PHASES[current_green_phase])
                    for _ in range(config.YELLOW_TIME):
                        if step >= config.SIMULATION_TIME:
                            break
                        traci.simulationStep()
                        collect_stats()
                        step += 1

                traci.trafficlight.setPhase('J1', chosen_phase)
                current_green_phase = chosen_phase
                time_in_phase = 0
                for _ in range(config.ACTION_COMMIT_TIME):
                    if step >= config.SIMULATION_TIME:
                        break
                    traci.simulationStep()
                    collect_stats()
                    step += 1
                    time_in_phase += 1

            # Use traci to get current phase for printing to avoid undefined errors
            curr_phase = traci.trafficlight.getPhase('J1')
            is_green = curr_phase % 2 == 0
            phase_name = config.GREEN_PHASES_NAME[curr_phase // 2] if is_green else config.GREEN_PHASES_NAME[curr_phase // 2] + " Yellow"
            print(f"Step {step} | Waiting Time: {step_stats['total_wait'][-1]:.2f} | Queue Length: {step_stats['queue_length'][-1]} | Flow Rate: {step_stats['flow_rate'][-1]:.2f} | Step Reward: {step_stats['step_reward'][-1]:.2f} | Phase: {phase_name} ({'Green' if is_green else 'Yellow'})")

        print("Simulation completed.")

    finally:
        if traci.isLoaded():
            traci.close()

    avg_wait, max_wait, min_wait, completed_vehicles, wait_times = get_stats_from_tripinfo(
        tripinfo_path)
    return cumulative_reward, avg_wait, max_wait, min_wait, completed_vehicles, step_stats, wait_times


def plot_evaluation_results(res_rl, res_base, experiment, results_dir):
    """Generates a comparison plot for the RL Agent vs Baseline."""
    plt.figure(figsize=(12, 6))

    # Subplot 1: Cumulative Reward
    plt.subplot(1, 2, 1)
    plt.plot(res_rl[5]['cumulative_reward'],
             label="RL Agent", color="blue", linewidth=2)
    plt.plot(res_base[5]['cumulative_reward'], label="Baseline",
             color="orange", linewidth=2, linestyle="--")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Reward: RL vs Baseline ({experiment})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Queue Length (Smooth)
    plt.subplot(1, 2, 2)
    window = 20
    def smooth(data): return np.convolve(
        data, np.ones(window)/window, mode='valid')

    rl_queue = smooth(res_rl[5]['queue_length'])
    base_queue = smooth(res_base[5]['queue_length'])

    plt.plot(range(window-1, len(res_rl[5]['queue_length'])),
             rl_queue, label="RL Agent", color="blue")
    plt.plot(range(window-1, len(res_base[5]['queue_length'])),
             base_queue, label="Baseline", color="orange", linestyle="--")
    plt.xlabel("Simulation Step")
    plt.ylabel("Queue Length (Halted Cars)")
    plt.title(f"Congestion: RL vs Baseline ({experiment})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"{experiment}_evaluation.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation plot saved to: {save_path}")


def plot_cumulative_waiting_time(waits_rl, waits_base, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(waits_rl, label="RL Agent", color="blue", linewidth=2)
    plt.plot(waits_base, label="Baseline", color="orange", linewidth=2)
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Waiting Time (s)")
    plt.title("Cumulative Waiting Time Comparison")
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
            ax.broken_barh([(start, end-start)], (phase, 0.8),
                           facecolors=colors.get(phase, 'gray'))
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

def plot_flow_rate(flow_rl, flow_base, save_path):
    """Plots the flow rate over time for both RL and Baseline."""
    plt.figure(figsize=(10, 6))
    plt.plot(flow_rl, label="RL Agent", color="blue", linewidth=2)
    plt.plot(flow_base, label="Baseline", color="orange", linewidth=2, linestyle="--")
    plt.xlabel("Simulation Step")
    plt.ylabel("Flow Rate (Vehicles/Step)")
    plt.title("Flow Rate Comparison (RL vs Baseline)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Flow rate plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="burst_spawn")
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
    RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../results"))

    experiments = [args.experiment] if args.experiment != "all" else [
        "burst_spawn", "periodic_uniform"]

    for exp in experiments:
        print(f"--- Evaluating Experiment: {exp} ---")
        model_dir = os.path.join(MODELS_DIR)
        
        # Check if model directory and files exist
        model_path = args.model
        if (not os.path.exists(model_path)):
            print("Model path provided does not exist", model_path)
            return

        if not model_path and os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
        
        if not model_path:
            print(f"No model folder or files found for {exp}, skipping RL evaluation.")
        else:
            print(f"Found model, path: {model_path}")

        trip_rl = os.path.join(RESULTS_DIR, exp, "trip_rl.xml")
        trip_base = os.path.join(RESULTS_DIR, exp, "trip_base.xml")
        os.makedirs(os.path.dirname(trip_rl), exist_ok=True)

        # Handle missing route file
        route_file = os.path.join(SCRIPT_DIR, "../sumo_config/hello.rou.xml")
        if not os.path.exists(route_file):
            print(f"Route file {route_file} missing. Generating new routes using generate_routes.py...")
            generate_routes(exp, route_file, bias_lane=0)
        else:
            # We must regenerate for each experiment to ensure traffic matches the experiment type
            # Defaulting to North (0) for evaluation, but could be randomized
            generate_routes(exp, route_file, bias_lane=0)

        print(f"Running Baseline Evaluation for {exp}...")
        res_base = run_evaluation(exp, None, args.nogui, True, trip_base)
        
        res_rl = res_base
        if model_path:
            print(f"Running RL Evaluation for {exp}...")
            res_rl = run_evaluation(exp, model_path, args.nogui, False, trip_rl)

        if res_base and res_base[5]:
            if res_rl and res_rl[5]:
                # Generate comparison plots
                plot_evaluation_results(
                    res_rl, res_base, exp, os.path.join(RESULTS_DIR, exp))

                plot_cumulative_waiting_time(res_rl[5]['total_wait'], res_base[5]['total_wait'],
                                             os.path.join(RESULTS_DIR, exp, f"{exp}_eval_cumulative_wait.png"))

                plot_waiting_time_distribution(res_rl[6], res_base[6],
                                               os.path.join(RESULTS_DIR, exp, f"{exp}_eval_wait_dist.png"))

                plot_phase_timeline(res_rl[5]['active_phase'], res_base[5]['active_phase'],
                                    os.path.join(RESULTS_DIR, exp, f"{exp}_eval_phase_timeline.png"))

                plot_flow_rate(res_rl[5]['flow_rate'], res_base[5]['flow_rate'],
                               os.path.join(RESULTS_DIR, exp, f"{exp}_eval_flow_rate.png"))

                print(f"Result for {exp}: RL Reward = {res_rl[0]:.2f}, Baseline Reward = {res_base[0]:.2f}")
            else:
                print(f"Result for {exp}: Baseline Reward = {res_base[0]:.2f} (RL skipped)")


if __name__ == "__main__":
    main()