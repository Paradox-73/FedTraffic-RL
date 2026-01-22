import traci
import sys

# Configuration
SUMO_CMD = ["sumo-gui", "-c", "hello.sumocfg", "--start"]

def get_state():
    """
    Member 1 Delivery: The 'Eyes' of the agent.
    Returns: [Queue_North, Queue_South, Queue_East, Queue_West]
    """
    q_N = traci.lane.getLastStepHaltingNumber("edge_N_in_0")
    q_S = traci.lane.getLastStepHaltingNumber("edge_S_in_0")
    q_E = traci.lane.getLastStepHaltingNumber("edge_E_in_0")
    q_W = traci.lane.getLastStepHaltingNumber("edge_W_in_0")
    return [q_N, q_S, q_E, q_W]

def get_reward():
    """
    Member 1 Delivery: The 'Score' of the agent.
    Returns: Negative total waiting time (Higher is better, 0 is perfect).
    """
    total_waiting_time = 0
    for v_id in traci.vehicle.getIDList():
        total_waiting_time += traci.vehicle.getWaitingTime(v_id)
    return -total_waiting_time

def run():
    traci.start(SUMO_CMD)
    print("🚀 Simulator Started with Heterogeneous Traffic!")

    step = 0
    while step < 1000:
        traci.simulationStep()
        
        # --- Collect Data for the AI ---
        current_state = get_state()
        current_reward = get_reward()

        # Print data so you can show Member 2 it works
        if step % 100 == 0:
            print(f"Step {step} | State: {current_state} | Reward: {current_reward}")

        step += 1

    traci.close()

if __name__ == "__main__":
    run()