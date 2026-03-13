import random
import config


def generate_routes(experiment_name, output_path):
    """
    Generates deterministic traffic flows for two stages:
    1. burst_spawn: High volume at t=0.
    2. periodic_uniform: Steady arrival throughout the simulation.
    """
    header = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vTypeDistribution id="mixed_traffic">
        <vType id="car" length="4.9" accel="3.0" decel="4.5" sigma="0.5" maxSpeed="13.89" color="yellow"/>
    </vTypeDistribution>

    <route id="NS" edges="edge_N_in edge_S_out"/>
    <route id="SN" edges="edge_S_in edge_N_out"/>
    <route id="EW" edges="edge_E_in edge_W_out"/>
    <route id="WE" edges="edge_W_in edge_E_out"/>
    <route id="NE" edges="edge_N_in edge_E_out"/>
    <route id="NW" edges="edge_N_in edge_W_out"/>
    <route id="SE" edges="edge_S_in edge_E_out"/>
    <route id="SW" edges="edge_S_in edge_W_out"/>
    <route id="EN" edges="edge_E_in edge_N_out"/>
    <route id="ES" edges="edge_E_in edge_S_out"/>
    <route id="WN" edges="edge_W_in edge_N_out"/>
    <route id="WS" edges="edge_W_in edge_S_out"/>
"""
    flows = ""
    total_vehicles = 0
    routes = ["NS", "SN", "EW", "WE", "NE",
              "NW", "SE", "SW", "EN", "ES", "WN", "WS"]

    if "burst_spawn" in experiment_name:
        # Stage 1: Spawn a fixed, large number of vehicles all at once at begin="0"
        # burst_count = 25
        for r in routes:
            flows += f'<flow id="burst_{r}" type="mixed_traffic" route="{r}" begin="0" end="1" number="{config.BURST_COUNT}"/>\n'
            total_vehicles += config.BURST_COUNT

    elif "periodic_uniform" in experiment_name:
        # Stage 2: Spawn vehicles periodically and uniformly across all lanes
        # Use period or vehsPerHour to ensure regular, evenly spaced intervals
        # period = 12
        for r in routes:
            flows += f'<flow id="periodic_{r}" type="mixed_traffic" route="{r}" begin="0" end="{config.SIMULATION_TIME}" period="{config.PERIOD}"/>\n'
            total_vehicles += (config.SIMULATION_TIME // config.PERIOD)

    with open(output_path, "w") as f:
        f.write(header + flows + "</routes>")

    return total_vehicles
