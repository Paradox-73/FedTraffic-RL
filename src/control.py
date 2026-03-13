import traci
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import TrafficLightAgent
from generate_routes import generate_routes
import argparse
import random
import time
import xml.etree.ElementTree as ET
import config

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = config.DEVICE
print("Using device:", DEVICE)

# --- Logger Setup ---
class Logger:
    def __init__(self, log_dir, experiment_name):
        self.terminal = sys.stdout
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        run_num = 1
        while os.path.exists(os.path.join(self.log_dir, f"run_{run_num}.log")):
            run_num += 1
        self.log_file_path = os.path.join(self.log_dir, f"run_{run_num}.log")
        self.log = open(self.log_file_path, "w", encoding="utf-8")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start(self):
        print(f"Terminal output also saved to: {self.log_file_path}")
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- State Function ---
def get_state(last_phase_time, current_phase):
    # State Dim = 18
    # 1. QUEUES
    q_north = np.clip(traci.edge.getLastStepHaltingNumber("edge_N_in") / 20.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber("edge_S_in") / 20.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber("edge_E_in") / 20.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber("edge_W_in") / 20.0, 0, 1)

    # 2. WAITS
    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 100.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 100.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 100.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 100.0, 0, 1)

    # 3. PHASE CONTEXT (One-Hot Encoding)
    is_n = 1 if current_phase == config.PHASE_N_GREEN else 0
    is_e = 1 if current_phase == config.PHASE_E_GREEN else 0
    is_s = 1 if current_phase == config.PHASE_S_GREEN else 0
    is_w = 1 if current_phase == config.PHASE_W_GREEN else 0

    # How long has it been green?
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

    # 4. PRESSURE
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber("edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 20.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber("edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 20.0, -1, 1)
    pressure_east = np.clip((traci.edge.getLastStepVehicleNumber("edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 20.0, -1, 1)
    pressure_west = np.clip((traci.edge.getLastStepVehicleNumber("edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 20.0, -1, 1)

    return np.array([q_north, q_south, q_east, q_west,
                     wait_north, wait_south, wait_east, wait_west,
                     is_n, is_e, is_s, is_w,
                     norm_last_phase_time,
                     pressure_north, pressure_south, pressure_east, pressure_west,
                     1.0])

def plot_rewards(rewards, losses, save_path):
    plt.figure(figsize=(12, 5))
    window = 10
    def moving_avg(data, window_size):
        if len(data) < window_size: return []
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    rewards_ma = moving_avg(rewards, window)
    losses_ma = moving_avg(losses, window)

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", alpha=0.3)
    if len(rewards_ma) > 0:
        plt.plot(range(window-1, len(rewards)), rewards_ma, label=f"MA (win={window})", color="blue", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards (Smoothed)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Raw Loss", color="orange", alpha=0.3)
    if len(losses_ma) > 0:
        plt.plot(range(window-1, len(losses)), losses_ma, label=f"MA (win={window})", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss (Smoothed)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_stats_from_tripinfo(filepath):
    if not os.path.exists(filepath): return 0, 0, 0, 0, []
    tree = ET.parse(filepath)
    root = tree.getroot()
    wait_times = [float(trip.get("waitingTime")) for trip in root.findall("tripinfo")]
    if not wait_times: return 0, 0, 0, 0, []
    return np.mean(wait_times), np.max(wait_times), np.min(wait_times), len(wait_times), wait_times

def run(experiment_name, args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    sumo_config_base = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config"))
    route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")
    sumo_log_file = os.path.join(sumo_config_base, "sumo_debug.log")
    tripinfo_path = os.path.join(sumo_config_base, "tripinfo.xml")

    sumo_bin = "sumo" if args.nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end",
                "--no-warnings", "--no-step-log", "--log", sumo_log_file,
                "--error-log", sumo_log_file, "--tripinfo-output", tripinfo_path]

    models_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../models/{experiment_name}"))
    results_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../results/{experiment_name}"))
    os.makedirs(models_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"model_{experiment_name}_epi_{config.NUM_OF_EPISODES}.pth")
    plot_path = os.path.join(results_dir, f"plot_{experiment_name}_epi_{config.NUM_OF_EPISODES}.png")
    logs_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))

    logger = Logger(logs_dir, experiment_name)
    logger.start()

    try:
        if traci.isLoaded(): traci.close()
        print(f"Starting experiment: {experiment_name}")

        agent = TrafficLightAgent(state_dim=18, action_dim=4)

        all_rewards = []; all_avg_losses = []
        all_avg_wait_times = []; all_completed_vehicles = []; all_total_vehicles = []
        epsilon = config.EPSILON_START

        for episode in range(config.NUM_OF_EPISODES):
            total_vehicles_in_sim = generate_routes(experiment_name, route_file_path)
            all_total_vehicles.append(total_vehicles_in_sim)
            time.sleep(0.1)

            if traci.isLoaded(): traci.close()
            traci.start(sumo_cmd)

            step = 0; episode_reward = 0; episode_losses = []
            time_in_phase = 0
            current_green_phase = traci.trafficlight.getPhase('J1')
            next_green_phase = current_green_phase
            state = get_state(time_in_phase, current_green_phase)

            # Initialize tracking for flow rate
            prev_outgoing_vehicles = {edge: set() for edge in config.OUTGOING_EDGES}

            while step < config.SIMULATION_TIME:
                current_phase = traci.trafficlight.getPhase('J1')

                if current_phase in config.YELLOW_PHASES.values():
                    traci.simulationStep()
                    
                    # Update tracking even during yellow to not miss vehicles
                    for edge in config.OUTGOING_EDGES:
                        prev_outgoing_vehicles[edge] = set(traci.edge.getLastStepVehicleIDs(edge))

                    step += 1
                    time_in_phase += 1
                    if time_in_phase >= config.YELLOW_TIME:
                        traci.trafficlight.setPhase('J1', next_green_phase)
                        current_green_phase = next_green_phase
                        time_in_phase = 0
                    continue

                if time_in_phase < config.MIN_GREEN_TIME:
                    action = config.GREEN_PHASES.index(current_green_phase)
                else:
                    action = agent.select_action(state, epsilon)

                chosen_phase = config.GREEN_PHASES[action]

                if chosen_phase != current_green_phase:
                    yellow_phase = config.YELLOW_PHASES[current_green_phase]
                    traci.trafficlight.setPhase('J1', yellow_phase)
                    next_green_phase = chosen_phase
                    time_in_phase = 0

                traci.simulationStep()
                step += 1
                time_in_phase += 1

                # --- NEW REWARD LOGIC (Flow Rate vs Waiting Cars) ---
                flow_count = 0
                for edge in config.OUTGOING_EDGES:
                    current_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
                    new_vehicles = current_vehicles - prev_outgoing_vehicles[edge]
                    flow_count += len(new_vehicles)
                    prev_outgoing_vehicles[edge] = current_vehicles
                
                # Flow rate for this step (time = 1)
                flow_rate = flow_count / 1.0 
                halting_cars = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in config.INCOMING_EDGES)
                reward = (config.W1 * flow_rate) - (config.W2 * halting_cars)

                next_phase_id = traci.trafficlight.getPhase('J1')
                next_state = get_state(time_in_phase, next_phase_id)
                
                agent.remember(state, action, reward, next_state, step >= config.SIMULATION_TIME)
                loss = agent.replay()
                if loss is not None: episode_losses.append(loss)
                state = next_state
                episode_reward += reward

            traci.close()

            avg_wait, _, _, completed_vehicles, _ = get_stats_from_tripinfo(tripinfo_path)
            all_avg_wait_times.append(avg_wait)
            all_completed_vehicles.append(completed_vehicles)

            if (episode + 1) % 2 == 0: agent.update_target_network()

            epsilon = max(config.EPSILON_MIN, epsilon * config.EPSILON_DECAY)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            all_rewards.append(episode_reward)
            all_avg_losses.append(avg_loss)
            print(f"Episode {episode+1}/{config.NUM_OF_EPISODES} | Reward: {episode_reward:7.2f} | Loss: {avg_loss:7.4f} | Epsilon: {epsilon:5.3f} | Avg Wait: {avg_wait:5.2f}s | Completed: {completed_vehicles}/{total_vehicles_in_sim}")

        torch.save(agent.model.state_dict(), model_path)
        plot_rewards(all_rewards, all_avg_losses, plot_path)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback; traceback.print_exc()
    finally:
        logger.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="burst_spawn", help="Experiment Name")
    parser.add_argument("--nogui", action="store_true", help="Run SUMO without GUI")
    args = parser.parse_args()

    if args.experiment == "all":
        stages = ["burst_spawn", "periodic_uniform"]
        for stage in stages:
            print(f"\nSTARTING EXPERIMENT: {stage}")
            run(stage, args)
    else:
        run(args.experiment, args)
