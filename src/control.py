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
import shutil
import time

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --- 8-PHASE SPLIT CYCLE CONSTANTS ---
# Cycle: North -> Yellow -> East -> Yellow -> South -> Yellow -> West -> Yellow
PHASE_N_GREEN = 0
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4
PHASE_S_YELLOW = 5
PHASE_W_GREEN = 6
PHASE_W_YELLOW = 7

MIN_GREEN_TIME = 10  # seconds
YELLOW_TIME = 3  # seconds

# Mapping from Agent Action (0-3) to Green Phase
GREEN_PHASES = [PHASE_N_GREEN, PHASE_E_GREEN, PHASE_S_GREEN, PHASE_W_GREEN]
# Mapping from Green Phase to its corresponding Yellow Phase
YELLOW_PHASES = {
    PHASE_N_GREEN: PHASE_N_YELLOW,
    PHASE_E_GREEN: PHASE_E_YELLOW,
    PHASE_S_GREEN: PHASE_S_YELLOW,
    PHASE_W_GREEN: PHASE_W_YELLOW,
}

NUM_OF_EPISODES = 10

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

# --- State & Reward Functions ---


def get_state(last_phase_time, current_phase):
    # State Dim = 18
    # Breakdown:
    #   4 (Queues)
    # + 4 (Waits)
    # + 4 (Phase One-Hot: [N, E, S, W])
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


def plot_rewards(rewards, losses, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Avg Loss", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_waits_comparison(waits_rl, waits_baseline, save_path):
    """Plot per-step total waiting time for RL vs fixed schedule."""
    steps = range(len(waits_rl))
    plt.figure(figsize=(12, 5))
    plt.plot(steps, waits_rl, label="RL Model", color="tab:blue")
    plt.plot(steps, waits_baseline,
             label="Fixed 90s Baseline", color="tab:orange")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Waiting Time (s)")
    plt.title("Intersection Performance: RL vs Fixed Cycle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run(experiment_name, args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sumo_config_base = os.path.normpath(
        os.path.join(SCRIPT_DIR, "../sumo_config"))
    route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")
    sumo_log_file = os.path.join(sumo_config_base, "sumo_debug.log")

    sumo_bin = "sumo" if args.nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end",
                "--no-warnings", "--no-step-log", "--log", sumo_log_file, "--error-log", sumo_log_file]

    models_dir = os.path.normpath(os.path.join(
        SCRIPT_DIR, f"../models/{experiment_name}"))
    results_dir = os.path.normpath(os.path.join(
        SCRIPT_DIR, f"../results/{experiment_name}"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(
        models_dir, f"model_{experiment_name}_epi_{NUM_OF_EPISODES}.pth")
    plot_path = os.path.join(
        results_dir, f"plot_{experiment_name}_epi_{NUM_OF_EPISODES}.png")

    logs_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))

    logger = Logger(logs_dir, experiment_name)
    logger.start()

    try:
        if traci.isLoaded():
            traci.close()
        print(f"Starting experiment: {experiment_name}")

        agent = TrafficLightAgent(state_dim=18, action_dim=4, device=DEVICE)

        all_rewards = []
        all_avg_losses = []
        epsilon = 1.0

        for episode in range(NUM_OF_EPISODES):
            generate_routes(experiment_name, route_file_path)
            time.sleep(0.1) # Small delay to ensure file is written before SUMO starts

            if traci.isLoaded():
                traci.close()
            traci.start(sumo_cmd)

            step = 0
            episode_reward = 0
            episode_losses = []

            last_action_time = 0
            time_in_phase = 0

            # The current green phase
            current_green_phase = traci.trafficlight.getPhase('J1')
            # The phase we are transitioning to
            next_green_phase = current_green_phase 

            state = get_state(time_in_phase, current_green_phase)

            previous_total_wait = 0

            while step < 1000:
                current_phase = traci.trafficlight.getPhase('J1')

                # If we are in a yellow phase, wait for it to finish
                if current_phase in YELLOW_PHASES.values():
                    traci.simulationStep()
                    step += 1
                    time_in_phase += 1
                    if time_in_phase >= YELLOW_TIME:
                        # Yellow finished, switch to target green
                        traci.trafficlight.setPhase('J1', next_green_phase)
                        current_green_phase = next_green_phase
                        time_in_phase = 0
                    continue

                # --- RL AGENT LOGIC ---
                # Only ask the agent for a decision if the min green time has passed
                if time_in_phase < MIN_GREEN_TIME:
                    action = GREEN_PHASES.index(
                        current_green_phase)  # Keep current phase
                else:
                    action = agent.select_action(state, epsilon)

                # Map agent action to a green phase
                chosen_phase = GREEN_PHASES[action]

                if chosen_phase == current_green_phase:
                    # Agent chose to continue the current green phase
                    pass
                else:
                    # Agent chose to switch, initiate yellow phase transition
                    yellow_phase = YELLOW_PHASES[current_green_phase]
                    traci.trafficlight.setPhase('J1', yellow_phase)
                    next_green_phase = chosen_phase
                    time_in_phase = 0  # Reset timer for yellow phase

                traci.simulationStep()
                step += 1
                time_in_phase += 1

                # --- REWARD & STATE UPDATE ---
                next_phase_id = traci.trafficlight.getPhase('J1')
                next_state = get_state(time_in_phase, next_phase_id)

                vehicle_ids = traci.vehicle.getIDList()
                current_total_wait = sum(
                    traci.vehicle.getWaitingTime(v_id) for v_id in vehicle_ids)

                reward = (previous_total_wait - current_total_wait) / 100.0
                previous_total_wait = current_total_wait

                agent.remember(state, action, reward, next_state, step >= 1000)

                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward

            if (episode + 1) % 10 == 0:
                agent.update_target_network()

            epsilon = max(0.01, epsilon * 0.995)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            all_rewards.append(episode_reward)
            all_avg_losses.append(avg_loss)
            print(f"  Episode {episode+1}/{NUM_OF_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {epsilon:5.3f}")

        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        plot_rewards(all_rewards, all_avg_losses, plot_path)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str,
                        default="stage1_baseline", help="Experiment Name")
    parser.add_argument("--nogui", action="store_true",
                        help="Run SUMO without GUI")

    args = parser.parse_args()

    if args.experiment == "all":
        stages = ["stage1_baseline", "stage2_rush_hour", "stage3_gridlock"]
        for stage in stages:
            print(f"\nSTARTING EXPERIMENT: {stage}")
            run(stage, args)
    else:
        run(args.experiment, args)
