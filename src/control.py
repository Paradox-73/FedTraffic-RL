import traci
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import TrafficLightAgent
import argparse
import random

# --- Configuration ---
# Build absolute paths based on the script's location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Open the log file with UTF-8 encoding to support all characters
        self.log = open(self.log_file_path, "w", encoding="utf-8")

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start(self):
        print(f"Terminal output is being saved to: {self.log_file_path}")
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log.close()
        print(f"\nLog file saved: {self.log_file_path}")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()




# --- State & Reward Functions ---
def get_state(last_phase_time, current_phase):
    q_north = traci.edge.getLastStepHaltingNumber("edge_N_in")
    q_south = traci.edge.getLastStepHaltingNumber("edge_S_in")
    q_east = traci.edge.getLastStepHaltingNumber("edge_E_in")
    q_west = traci.edge.getLastStepHaltingNumber("edge_W_in")
    wait_north = traci.edge.getWaitingTime("edge_N_in")
    wait_south = traci.edge.getWaitingTime("edge_S_in")
    wait_east = traci.edge.getWaitingTime("edge_E_in")
    wait_west = traci.edge.getWaitingTime("edge_W_in")
    is_ns_green = 1 if current_phase == 0 else 0
    return np.array([q_north, q_south, q_east, q_west, wait_north, wait_south, wait_east, wait_west, is_ns_green, last_phase_time, 1.0])

# --- Plotting Function ---
def plot_rewards(rewards, losses, plot_path):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, label='Rewards')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Loss', color=color)
    ax2.plot(losses, color=color, label='Losses')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('DQN Training Progress')
    plt.savefig(plot_path)
    print(f"\nTraining plot saved to {plot_path}")

# --- Main Training Loop ---

def run(experiment_name):
    # Seeding for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Dynamic Path Configuration ---
    sumo_config_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../sumo_config/{experiment_name}"))
    os.makedirs(sumo_config_dir, exist_ok=True)
    


    sumo_cfg_path = os.path.join(sumo_config_dir, "hello.sumocfg")
    sumo_cmd = ["sumo-gui", "-c", sumo_cfg_path, "--start", "--quit-on-end"]
    
    models_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../models/{experiment_name}"))
    results_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../results/{experiment_name}"))
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "dqn_traffic_model.pth")
    plot_path = os.path.join(results_dir, "training_progress.png")
    
    logs_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))
    logger = Logger(logs_dir, experiment_name)
    logger.start()

    try:
        # Initial check to ensure any previous SUMO instance is closed.
        if traci.isLoaded():
            traci.close()
        
        print(f"Simulator Started for experiment: {experiment_name} on {DEVICE}")
        
        agent = TrafficLightAgent(state_dim=11, action_dim=2, device=DEVICE)
        num_episodes = 200
        max_steps_per_episode = 1000
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        target_update_freq = 10
        epsilon = epsilon_start
        all_rewards = []
        all_avg_losses = []

        for episode in range(num_episodes):
            # Start a new SUMO simulation for each episode to ensure a fresh environment
            # including vehicle generation from flows defined in hello.rou.xml.
            try:
                if traci.isLoaded(): # Ensure previous episode's SUMO is closed
                    traci.close()
                traci.start(sumo_cmd)
                # Ensure context subscription for reward calculation (if needed, otherwise rely on direct calls)
                # Removed traci.simulation.subscribeContext as it was causing issues.
                # Direct calls to traci.edge.getWaitingTime and traci.vehicle.getIDList are used instead.

                step, last_phase_time, episode_reward = 0, 0, 0
                episode_losses = []
                current_phase = traci.trafficlight.getPhase('J1')
                state = get_state(last_phase_time, current_phase)

                while step < max_steps_per_episode:
                    traci.simulationStep()
                    action = agent.select_action(state, epsilon)
                    if action == 1:
                        # Toggle between NS-Green (0) and EW-Green (2)
                        new_phase = 2 if current_phase == 0 else 0
                        traci.trafficlight.setPhase('J1', new_phase)
                        last_phase_time = 0
                    
                    current_phase = traci.trafficlight.getPhase('J1')
                    # Ensure there are vehicles before trying to sum their waiting times
                    vehicle_ids = traci.vehicle.getIDList()
                    if vehicle_ids:
                        reward = -sum(traci.vehicle.getWaitingTime(v_id) for v_id in vehicle_ids)
                    else:
                        reward = 0.0 # No vehicles, no waiting time
                    next_state = get_state(last_phase_time, current_phase)
                    agent.remember(state, action, reward, next_state, step >= max_steps_per_episode)
                    loss = agent.replay()
                    if loss is not None: episode_losses.append(loss)
                    state = next_state
                    episode_reward += reward
                    step += 1
                    last_phase_time += 1

                if (episode + 1) % target_update_freq == 0: agent.update_target_network()
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                all_rewards.append(episode_reward)
                all_avg_losses.append(avg_loss)
                print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f}")

            except traci.TraCIException as e:
                print(f"\nA SUMO error occurred in episode {episode+1}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"\nAn unexpected error occurred in episode {episode+1}: {e}", file=sys.stderr)
            finally:
                if traci.isLoaded():
                    traci.close()

        torch.save(agent.model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        plot_rewards(all_rewards, all_avg_losses, plot_path)

    except Exception as e: # Catch any errors from outside the episode loop
        print(f"\nAn unexpected error occurred during experiment setup: {e}", file=sys.stderr)
    finally:
        logger.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL traffic control simulation.")
    parser.add_argument("--experiment", type=str, default="stage0_baseline", 
                        help="Name of the experiment to run, or 'all'.")
    args = parser.parse_args()

    if args.experiment == "all":
        # Note: This assumes the script is run from the project root or that paths are correctly structured.
        experiments_dir = os.path.join(SCRIPT_DIR, "../sumo_config")
        # Filter for directories that start with 'stage'
        all_experiments = sorted([
            d for d in os.listdir(experiments_dir) 
            if os.path.isdir(os.path.join(experiments_dir, d)) and d.startswith('stage')
        ])
        for exp in all_experiments:
            print(f"\n----- Running Experiment: {exp} બા-----")
            run(exp)
        print("\n----- All experiments completed -----")
    else:
        run(args.experiment)
