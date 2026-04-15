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
from collections import deque

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = config.DEVICE
print("Using device:", DEVICE)

# --- Logger Setup ---
class Logger:
    def __init__(self, log_dir, experiment_name):
        self.terminal = sys.stdout
        self.base_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.base_log_dir, exist_ok=True)

        run_num = 1
        while os.path.exists(os.path.join(self.base_log_dir, f"run_{run_num}")):
            run_num += 1
        self.run_dir = os.path.join(self.base_log_dir, f"run_{run_num}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def set_log_file(self, filename):
        if self.log_file:
            self.log_file.close()
        log_path = os.path.join(self.run_dir, filename)
        self.log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file:
            self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()


# ---------------------------------------------------------------------------
# State Function  (state_dim = 22)
# NEW: accepts time_since_green dict  →  appends 4 starvation features
# ---------------------------------------------------------------------------
def get_state(last_phase_time, current_phase, time_since_green):
    """
    Returns a 22-dimensional state vector.

    Breakdown:
      [0-3]   Queue lengths per incoming edge          (4)
      [4-7]   Waiting times per incoming edge           (4)
      [8-11]  One-hot current green phase               (4)
      [12]    Normalised time in current phase           (1)
      [13-16] Pressure (in - out) per direction         (4)
      [17]    Bias feature (constant 1.0)               (1)
      [18-21] Starvation: steps since each dir had green (4)  ← NEW
    """
    # 1. QUEUES
    q_north = np.clip(traci.edge.getLastStepHaltingNumber("edge_N_in") / 60.0, 0, 1)
    q_south = np.clip(traci.edge.getLastStepHaltingNumber("edge_S_in") / 60.0, 0, 1)
    q_east  = np.clip(traci.edge.getLastStepHaltingNumber("edge_E_in") / 60.0, 0, 1)
    q_west  = np.clip(traci.edge.getLastStepHaltingNumber("edge_W_in") / 60.0, 0, 1)

    # 2. WAITS
    wait_north = np.clip(traci.edge.getWaitingTime("edge_N_in") / 2000.0, 0, 1)
    wait_south = np.clip(traci.edge.getWaitingTime("edge_S_in") / 2000.0, 0, 1)
    wait_east  = np.clip(traci.edge.getWaitingTime("edge_E_in") / 2000.0, 0, 1)
    wait_west  = np.clip(traci.edge.getWaitingTime("edge_W_in") / 2000.0, 0, 1)

    # 3. PHASE ONE-HOT
    is_n = 1 if current_phase == config.PHASE_N_GREEN else 0
    is_e = 1 if current_phase == config.PHASE_E_GREEN else 0
    is_s = 1 if current_phase == config.PHASE_S_GREEN else 0
    is_w = 1 if current_phase == config.PHASE_W_GREEN else 0

    # 4. TIME IN CURRENT PHASE
    norm_last_phase_time = np.clip(last_phase_time / 150.0, 0, 1)

    # 5. PRESSURE
    pressure_north = np.clip((traci.edge.getLastStepVehicleNumber("edge_N_in") - traci.edge.getLastStepVehicleNumber("edge_S_out")) / 60.0, -1, 1)
    pressure_south = np.clip((traci.edge.getLastStepVehicleNumber("edge_S_in") - traci.edge.getLastStepVehicleNumber("edge_N_out")) / 60.0, -1, 1)
    pressure_east  = np.clip((traci.edge.getLastStepVehicleNumber("edge_E_in") - traci.edge.getLastStepVehicleNumber("edge_W_out")) / 60.0, -1, 1)
    pressure_west  = np.clip((traci.edge.getLastStepVehicleNumber("edge_W_in") - traci.edge.getLastStepVehicleNumber("edge_E_out")) / 60.0, -1, 1)

    # 6. STARVATION (NEW)
    # How long has each direction been waiting for green, normalised to [0, 1].
    # When all four values are high, the agent knows traffic is uniform and
    # must rotate rather than lock onto a single phase.
    starve_n = np.clip(time_since_green[config.PHASE_N_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_e = np.clip(time_since_green[config.PHASE_E_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_s = np.clip(time_since_green[config.PHASE_S_GREEN] / config.STARVATION_NORM, 0, 1)
    starve_w = np.clip(time_since_green[config.PHASE_W_GREEN] / config.STARVATION_NORM, 0, 1)

    return np.array([
        q_north, q_south, q_east, q_west,
        wait_north, wait_south, wait_east, wait_west,
        is_n, is_e, is_s, is_w,
        norm_last_phase_time,
        pressure_north, pressure_south, pressure_east, pressure_west,
        1.0,
        starve_n, starve_e, starve_s, starve_w,   # indices 18-21
    ], dtype=np.float32)


def plot_rewards(rewards, losses, save_path):
    plt.figure(figsize=(12, 5))
    window = 10

    def moving_avg(data, window_size):
        if len(data) < window_size:
            return []
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    rewards_ma = moving_avg(rewards, window)
    losses_ma  = moving_avg(losses,  window)

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", alpha=0.3)
    if len(rewards_ma) > 0:
        plt.plot(range(window - 1, len(rewards)), rewards_ma,
                 label=f"MA (win={window})", color="blue", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.title("Training Rewards (Smoothed)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Raw Loss", color="orange", alpha=0.3)
    if len(losses_ma) > 0:
        plt.plot(range(window - 1, len(losses)), losses_ma,
                 label=f"MA (win={window})", color="red", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Loss")
    plt.title("Training Loss (Smoothed)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_step_metrics(step_data, save_path):
    steps        = [d['step']         for d in step_data]
    waiting_times = [d['waiting_time'] for d in step_data]
    flow_rates   = [d['flow_rate']    for d in step_data]
    halting_cars = [d['halting_cars'] for d in step_data]

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(steps, waiting_times, label="Total Waiting Time (s)")
    plt.ylabel("Waiting Time"); plt.legend(); plt.grid(True)
    plt.title("Step-wise Metrics")

    plt.subplot(3, 1, 2)
    plt.plot(steps, flow_rates, label="Flow Rate (veh/step)", color='green')
    plt.ylabel("Flow Rate"); plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(steps, halting_cars, label="Halting Cars", color='red')
    plt.xlabel("Step"); plt.ylabel("Halting Cars")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def collect_step_metrics(prev_outgoing_vehicles):
    flow_count = 0
    for edge in config.OUTGOING_EDGES:
        current_vehicles = set(traci.edge.getLastStepVehicleIDs(edge))
        new_vehicles = current_vehicles - prev_outgoing_vehicles[edge]
        flow_count += len(new_vehicles)
        prev_outgoing_vehicles[edge] = current_vehicles

    flow_rate        = flow_count / 1.0
    halting_cars     = sum(traci.edge.getLastStepHaltingNumber(e) for e in config.INCOMING_EDGES)
    total_wait_time  = sum(traci.edge.getWaitingTime(e)           for e in config.INCOMING_EDGES)
    return flow_rate, halting_cars, total_wait_time, prev_outgoing_vehicles


def get_stats_from_tripinfo(filepath):
    if not os.path.exists(filepath):
        return 0, 0, 0, 0, []
    tree = ET.parse(filepath)
    root = tree.getroot()
    wait_times = [float(trip.get("waitingTime")) for trip in root.findall("tripinfo")]
    if not wait_times:
        return 0, 0, 0, 0, []
    return np.mean(wait_times), np.max(wait_times), np.min(wait_times), len(wait_times), wait_times


def run(experiment_name, args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    sumo_config_base = os.path.normpath(os.path.join(SCRIPT_DIR, "../sumo_config"))
    route_file_path  = os.path.join(sumo_config_base, "hello.rou.xml")
    sumo_cfg_path    = os.path.join(sumo_config_base, "hello.sumocfg")
    sumo_log_file    = os.path.join(sumo_config_base, "sumo_debug.log")
    tripinfo_path    = os.path.join(sumo_config_base, "tripinfo.xml")

    sumo_bin = "sumo" if args.nogui else "sumo-gui"
    sumo_cmd = [sumo_bin, "-c", sumo_cfg_path, "--start", "--quit-on-end",
                "--no-warnings", "--no-step-log", "--log", sumo_log_file,
                "--error-log", sumo_log_file, "--tripinfo-output", tripinfo_path]

    models_dir  = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
    results_dir = os.path.normpath(os.path.join(SCRIPT_DIR, f"../results/{experiment_name}"))
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir,  f"model_{experiment_name}_epi_{config.NUM_OF_EPISODES}.pth")
    plot_path  = os.path.join(results_dir, f"plot_{experiment_name}_epi_{config.NUM_OF_EPISODES}.png")
    logs_dir   = os.path.normpath(os.path.join(SCRIPT_DIR, "../logs"))

    logger = Logger(logs_dir, experiment_name)
    logger.set_log_file("setup.log")

    try:
        if traci.isLoaded():
            traci.close()
        print(f"Starting experiment: {experiment_name}")

        agent = TrafficLightAgent(state_dim=22, action_dim=4)

        all_rewards           = []
        all_avg_losses        = []
        all_avg_wait_times    = []
        all_completed_vehicles = []
        all_total_vehicles    = []
        epsilon = config.EPSILON_START

        for episode in range(config.NUM_OF_EPISODES):
            logger.set_log_file(f"episode_{episode+1}.log")

            # ------------------------------------------------------------------
            # Route generation
            # For one_heavy_three_light: inject a periodic_uniform episode every
            # MIX_UNIFORM_EVERY_N episodes so the agent trains on uniform traffic.
            # ------------------------------------------------------------------
            bias_lane = episode % 4
            if "one_heavy_three_light" in experiment_name and \
               (episode + 1) % config.MIX_UNIFORM_EVERY_N == 0:
                route_type = "periodic_uniform"
                print(f"  [Episode {episode+1}] Mixed-in periodic_uniform episode")
            else:
                route_type = experiment_name

            total_vehicles_in_sim = generate_routes(route_type, route_file_path, bias_lane=bias_lane)
            all_total_vehicles.append(total_vehicles_in_sim)
            time.sleep(0.1)

            if traci.isLoaded():
                traci.close()
            traci.start(sumo_cmd)

            step              = 0
            episode_reward    = 0
            episode_losses    = []
            time_in_phase     = 0
            current_green_phase = traci.trafficlight.getPhase('J1')
            next_green_phase    = current_green_phase

            # ------------------------------------------------------------------
            # Starvation tracker (NEW)
            # Counts how many steps each direction has gone without a green phase.
            # Reset to 0 the moment a direction receives green.
            # ------------------------------------------------------------------
            time_since_green = {p: 0 for p in config.GREEN_PHASES}
            # The initial phase already has green, so start its counter at 0
            time_since_green[current_green_phase] = 0

            state = get_state(time_in_phase, current_green_phase, time_since_green)

            prev_outgoing_vehicles = {edge: set() for edge in config.OUTGOING_EDGES}
            step_metrics_log = []
            flow_window = deque(maxlen=10)

            while step < config.SIMULATION_TIME:
                current_phase = traci.trafficlight.getPhase('J1')

                # --------------------------------------------------------------
                # Yellow phase: step through it, then switch to chosen green
                # --------------------------------------------------------------
                if current_phase in config.YELLOW_PHASES.values():
                    traci.simulationStep()
                    step += 1
                    time_in_phase += 1

                    flow_rate, halting_cars, total_waiting_time, prev_outgoing_vehicles = \
                        collect_step_metrics(prev_outgoing_vehicles)
                    flow_window.append(flow_rate)
                    smoothed_flow = sum(flow_window) / len(flow_window)

                    step_metrics_log.append({
                        'step': step,
                        'waiting_time': total_waiting_time,
                        'flow_rate':    flow_rate,
                        'halting_cars': halting_cars,
                    })

                    if not args.nogui:
                        print(f"Step: {step} | Waiting: {total_waiting_time:7.2f} | "
                              f"Smoothed Flow: {smoothed_flow:4.2f} | Halting: {halting_cars}")

                    if time_in_phase >= config.YELLOW_TIME:
                        traci.trafficlight.setPhase('J1', next_green_phase)
                        current_green_phase = next_green_phase
                        time_since_green[current_green_phase] = 0   # reset on green
                        time_in_phase = 0
                    continue

                # --------------------------------------------------------------
                # Green phase: enforce minimum, then let agent decide
                # --------------------------------------------------------------
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

                # --------------------------------------------------------------
                # Update starvation counters
                # The current green phase resets to 0; all others increment.
                # --------------------------------------------------------------
                for p in config.GREEN_PHASES:
                    if p == current_green_phase:
                        time_since_green[p] = 0
                    else:
                        time_since_green[p] += 1

                flow_rate, halting_cars, total_waiting_time, prev_outgoing_vehicles = \
                    collect_step_metrics(prev_outgoing_vehicles)
                flow_window.append(flow_rate)
                smoothed_flow = sum(flow_window) / len(flow_window)

                step_metrics_log.append({
                    'step': step,
                    'waiting_time': total_waiting_time,
                    'flow_rate':    flow_rate,
                    'halting_cars': halting_cars,
                })

                # --------------------------------------------------------------
                # Reward
                # Sum the starvation of all *non-current* directions, normalised.
                # This directly penalises locking onto one phase for too long.
                # --------------------------------------------------------------
                starvation_penalty = sum(
                    time_since_green[p]
                    for p in config.GREEN_PHASES
                    if p != current_green_phase
                ) / config.STARVATION_NORM

                reward = (config.W1 * smoothed_flow) - (config.W2 * halting_cars) - (config.W3 * starvation_penalty)

                next_phase_id = traci.trafficlight.getPhase('J1')
                next_state    = get_state(time_in_phase, next_phase_id, time_since_green)

                agent.remember(state, action, reward, next_state, step >= config.SIMULATION_TIME)
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward

            traci.close()

            step_plot_path = os.path.join(
                results_dir, f"step_metrics_{experiment_name}_epi_{episode+1}.png")
            plot_step_metrics(step_metrics_log, step_plot_path)

            avg_wait, _, _, completed_vehicles, _ = get_stats_from_tripinfo(tripinfo_path)
            all_avg_wait_times.append(avg_wait)
            all_completed_vehicles.append(completed_vehicles)

            if (episode + 1) % 2 == 0:
                agent.update_target_network()

            epsilon = max(config.EPSILON_MIN, epsilon * config.EPSILON_DECAY)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            all_rewards.append(episode_reward)
            all_avg_losses.append(avg_loss)
            print(f"Episode {episode+1}/{config.NUM_OF_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {epsilon:5.3f} | Avg Wait: {avg_wait:5.2f}s | "
                  f"Completed: {completed_vehicles}/{total_vehicles_in_sim} | "
                  f"RouteType: {route_type}")

        torch.save(agent.model.state_dict(), model_path)
        plot_rewards(all_rewards, all_avg_losses, plot_path)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback; traceback.print_exc()
    finally:
        logger.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="burst_spawn")
    parser.add_argument("--nogui", action="store_true")
    args = parser.parse_args()

    if args.experiment == "all":
        for stage in ["burst_spawn", "periodic_uniform", "one_heavy_three_light"]:
            print(f"\nSTARTING EXPERIMENT: {stage}")
            run(stage, args)
    else:
        run(args.experiment, args)