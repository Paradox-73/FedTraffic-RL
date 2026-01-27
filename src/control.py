import traci
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import TrafficLightAgent

# --- Configuration ---
# Build absolute paths based on the script's location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config/hello.sumocfg"))

SUMO_CMD = ["sumo-gui", "-c", SUMO_CFG_PATH, "--start", "--quit-on-end"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define project root-level directories
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../results"))
LOGS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))

# Define full paths for output files
MODEL_PATH = os.path.join(MODELS_DIR, "dqn_traffic_model.pth")
PLOT_PATH = os.path.join(RESULTS_DIR, "training_progress.png")

# --- Logger Setup ---
class Logger:
    def __init__(self, log_dir):
        self.terminal = sys.stdout
        self.log_dir = log_dir
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
    is_ns_green = 1 if current_phase in [0, 1] else 0
    is_yellow = 1 if current_phase in [1, 3] else 0
    return np.array([q_north, q_south, q_east, q_west, wait_north, wait_south, wait_east, wait_west, is_ns_green, last_phase_time, is_yellow, 1.0])

# --- Plotting Function ---
def plot_rewards(rewards, losses):
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
    plt.savefig(PLOT_PATH)
    print(f"\nTraining plot saved to {PLOT_PATH}")

# --- Main Training Loop ---
def run():
    traci.start(SUMO_CMD)
    print(f"Simulator Started on {DEVICE}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    agent = TrafficLightAgent(state_dim=12, action_dim=2, device=DEVICE)
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
        traci.load(["-c", SUMO_CFG_PATH, "--start"])
        step, last_phase_time, episode_reward = 0, 0, 0
        episode_losses = []
        current_phase = traci.trafficlight.getPhase('J1')
        state = get_state(last_phase_time, current_phase)

        while step < max_steps_per_episode:
            traci.simulationStep()
            action = agent.select_action(state, epsilon)
            if action == 1:
                traci.trafficlight.setPhase('J1', (current_phase + 1) % 4)
                for _ in range(5):
                    if step < max_steps_per_episode: traci.simulationStep(); step += 1
                traci.trafficlight.setPhase('J1', (current_phase + 2) % 4)
                last_phase_time = 0
            
            current_phase = traci.trafficlight.getPhase('J1')
            reward = -sum(traci.vehicle.getWaitingTime(v_id) for v_id in traci.vehicle.getIDList())
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

    torch.save(agent.model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    plot_rewards(all_rewards, all_avg_losses)
    traci.close()

if __name__ == "__main__":
    logger = Logger(LOGS_DIR)
    logger.start()
    try:
        run()
    except traci.TraCIException as e:
        print(f"\nA SUMO error occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if traci.isLoaded():
            traci.close()
        logger.stop()
