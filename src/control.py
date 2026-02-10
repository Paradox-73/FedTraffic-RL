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

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Traffic Light Phases for J1 (from hello.net.xml)
PHASE_NS_GREEN_ID = 0
PHASE_NS_YELLOW_ID = 1
PHASE_EW_GREEN_ID = 2
PHASE_EW_YELLOW_ID = 3

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
        print(f"Terminal output saved to: {self.log_file_path}")
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
    # State Dim = 15
    # breakdown: 4 (queues) + 4 (waits) + 2 (phase Info) + 4 (pressure) + 1 (bias) = 15

    # QUEUE LENGTHS
    # traci.edge.getLastStepHaltingNumber returns the count of cars with speed < 0.1 m/s.
    # divide by 20.0 because ~20 cars is roughly a full lane (approx 100m).
    q_north = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_N_in") / 20.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_S_in") / 20.0, 0, 1)
    q_east = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_E_in") / 20.0, 0, 1)
    q_west = np.clip(traci.edge.getLastStepHaltingNumber(
        "edge_W_in") / 20.0, 0, 1)

    # CUMULATIVE WAIT TIME
    # traci.edge.getWaitingTime returns the SUM of waiting seconds for ALL cars on that edge.
    # divide by 100.0 to normalize.
    wait_north = np.clip(traci.edge.getWaitingTime(
        "edge_N_in") / 100.0, 0, 1)  # 100 is a softcap
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 100.0, 0, 1)
    wait_east = np.clip(traci.edge.getWaitingTime("edge_E_in") / 100.0, 0, 1)
    wait_west = np.clip(traci.edge.getWaitingTime("edge_W_in") / 100.0, 0, 1)

    # PHASE CONTEXT
    # one-Hot Encoding: 1 if North-South is Green, 0 otherwise.
    is_ns_green = 1 if current_phase == PHASE_NS_GREEN_ID else 0

    # how long has the light been green?
    # normalize by 150 seconds (assumed max reasonable cycle time).
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

    # PRESSURE (The "Throughput" Metric)
    # pressure = Incoming Vehicles - outgoing Vehicles.
    # high +pressure: cars are piling up at the input but not leaving (bottle neck).
    # high -pressure: cars are leaving faster than they arrive (free flow).
    # normaliz by 20.0 to match the queue scaling.
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 20.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 20.0, -1, 1)
    pressure_east = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 20.0, -1, 1)
    pressure_west = np.clip((traci.edge.getLastStepVehicleNumber(
        "edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 20.0, -1, 1)

    # ASSEMBLE VECTOR
    # The final "1.0" is the Bias Node (keeps neurons active even if all inputs are 0).
    return np.array([q_north, q_south, q_east, q_west,
                     wait_north, wait_south, wait_east, wait_west,
                     is_ns_green, norm_last_phase_time,
                     pressure_north, pressure_south, pressure_east, pressure_west,
                     1.0])


def plot_rewards(rewards, losses, plot_path):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Rewards')
    if losses:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Loss', color=color)
        ax2.plot(losses, color=color, label='Losses')
    plt.title('DQN Training Progress')
    plt.savefig(plot_path)
    plt.close()


def run(experiment_name, args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate routes directly in the base config directory
    sumo_config_base = os.path.normpath(
        os.path.join(SCRIPT_DIR, "../sumo_config"))
    route_file_path = os.path.join(sumo_config_base, "hello.rou.xml")
    sumo_cfg_path = os.path.join(sumo_config_base, "hello.sumocfg")

    # Update sumo_cmd to use the debug log in the base directory
    sumo_log_file = os.path.join(sumo_config_base, "sumo_debug.log")

    sumo_bin = "sumo" if args.nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end",
                "--no-warnings", "--log", sumo_log_file, "--error-log", sumo_log_file]

    models_dir = os.path.normpath(os.path.join(
        SCRIPT_DIR, f"../models/{experiment_name}"))
    results_dir = os.path.normpath(os.path.join(
        SCRIPT_DIR, f"../results/{experiment_name}"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "dqn_traffic_model.pth")
    plot_path = os.path.join(results_dir, "training_progress.png")

    logs_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))
    logger = Logger(logs_dir, experiment_name)
    logger.start()

    try:
        if traci.isLoaded():
            traci.close()
        print(
            f"Simulator Started for experiment: {experiment_name} on {DEVICE}")

        # State Dim 15 (Matches get_state, after adding pressure)
        agent = TrafficLightAgent(state_dim=15, action_dim=2, device=DEVICE)

        all_rewards = []
        all_avg_losses = []
        epsilon = 1.0

        num_episodes = 100  # Default number of episodes
        for episode in range(num_episodes):
            # GENERATE NEW ROUTES EVERY EPISODE
            generate_routes(experiment_name, route_file_path)

            if traci.isLoaded():
                traci.close()
            traci.start(sumo_cmd)

            step = 0
            last_phase_time = 0
            episode_reward = 0
            episode_losses = []
            previous_total_wait = 0  # Initialize for differential reward

            state = get_state(
                last_phase_time, traci.trafficlight.getPhase('J1'))

            while step < 1000:
                current_phase_id = traci.trafficlight.getPhase('J1')

                # Agent only acts on green phases
                if current_phase_id == PHASE_NS_GREEN_ID or current_phase_id == PHASE_EW_GREEN_ID:
                    action = agent.select_action(state, epsilon)
                    if action == 1:
                        traci.trafficlight.setPhase('J1', current_phase_id + 1)
                        last_phase_time = 0
                else:
                    action = 0  # Force "stay" during yellow

                traci.simulationStep()
                step += 1
                last_phase_time += 1

                next_phase_id = traci.trafficlight.getPhase('J1')
                next_state = get_state(last_phase_time, next_phase_id)

                # Reward calculation
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
            print(
                f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {epsilon:.3f}")

        torch.save(agent.model.state_dict(), model_path)
        plot_rewards(all_rewards, all_avg_losses, plot_path)

    except Exception as e:
        print(f"Fatal error: {e}")
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
            print(f"\n🚀 STARTING EXPERIMENT: {stage} 🚀")
            run(stage, args)
    else:
        run(args.experiment, args)
