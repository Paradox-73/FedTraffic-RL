import random


def generate_routes(experiment_name, output_path):
    """
    Generates the hello.rou.xml file and returns the total number of vehicles.
    This file tells SUMO:
    1. What kind of vehicles exist (Cars, Buses, Motos).
    2. Where the roads go (Routes).
    3. When vehicles appear (Flows).
    """
    total_vehicles = 0

    # --- XML HEADER & VEHICLE DEFINITIONS ---
    header = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vTypeDistribution id="mixed_traffic">
        <vType id="car" length="4.9" accel="2.3" maxSpeed="15.0" color="yellow" probability="0.80" speedFactor="0.96" speedDev="0.1"/>
        <vType id="bus" length="12.0" accel="1.0" maxSpeed="10.0" color="red" probability="0.10" speedFactor="0.9"/>
        <vType id="moto" length="2.0" accel="3.0" maxSpeed="20.0" color="blue" probability="0.10" speedFactor="1.1"/>
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
    # --- STAGE 1 LOGIC: The Baseline ---
    if "stage1" in experiment_name:
        base_prob = random.uniform(0.05, 0.1)
        for _ in range(4):  # 4 straight routes
            total_vehicles += 1000 * (base_prob * 0.8)
        for _ in range(4):  # 4 turning routes
            total_vehicles += 1000 * (base_prob * 0.2)

        flows += f'<flow id="f_NS" type="mixed_traffic" route="NS" begin="0" end="1000" probability="{base_prob * 0.8}"/>\n'
        flows += f'<flow id="f_SN" type="mixed_traffic" route="SN" begin="0" end="1000" probability="{base_prob * 0.8}"/>\n'
        flows += f'<flow id="f_EW" type="mixed_traffic" route="EW" begin="0" end="1000" probability="{base_prob * 0.8}"/>\n'
        flows += f'<flow id="f_WE" type="mixed_traffic" route="WE" begin="0" end="1000" probability="{base_prob * 0.8}"/>\n'
        flows += f'<flow id="f_NW" type="mixed_traffic" route="NW" begin="0" end="1000" probability="{base_prob * 0.2}"/>\n'
        flows += f'<flow id="f_SE" type="mixed_traffic" route="SE" begin="0" end="1000" probability="{base_prob * 0.2}"/>\n'
        flows += f'<flow id="f_EN" type="mixed_traffic" route="EN" begin="0" end="1000" probability="{base_prob * 0.2}"/>\n'
        flows += f'<flow id="f_WS" type="mixed_traffic" route="WS" begin="0" end="1000" probability="{base_prob * 0.2}"/>\n'

    # --- STAGE 2 LOGIC: Rush Hour ---
    elif "stage2" in experiment_name:
        peak_start = random.randint(200, 400)
        peak_end = peak_start + 400
        lull_prob = 0.05
        ns_rush_prob = 0.15
        ew_rush_prob = 0.0625
        cool_prob = 0.05

        total_vehicles += 4 * (peak_start * lull_prob)
        total_vehicles += 2 * ((peak_end - peak_start) * ns_rush_prob)
        total_vehicles += 2 * ((peak_end - peak_start) * ew_rush_prob)
        total_vehicles += 4 * ((1000 - peak_end) * cool_prob)

        flows += f'<flow id="lull_NS" type="mixed_traffic" route="NS" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_SN" type="mixed_traffic" route="SN" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_EW" type="mixed_traffic" route="EW" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_WE" type="mixed_traffic" route="WE" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="rush_NS" type="mixed_traffic" route="NS" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_SN" type="mixed_traffic" route="SN" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_EW" type="mixed_traffic" route="EW" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'
        flows += f'<flow id="rush_WE" type="mixed_traffic" route="WE" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'
        flows += f'<flow id="cool_NS" type="mixed_traffic" route="NS" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_SN" type="mixed_traffic" route="SN" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_EW" type="mixed_traffic" route="EW" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_WE" type="mixed_traffic" route="WE" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'

    # --- STAGE 3 LOGIC: Gridlock ---
    elif "stage3" in experiment_name:
        prob = random.uniform(0.05, 0.1)
        for _ in range(4): # 4 origins
            total_vehicles += 1000 * (prob * 0.5)  # Straight
            total_vehicles += 1000 * (prob * 0.25) # Right
            total_vehicles += 1000 * (prob * 0.25) # Left

        for origin, dests in [("N", ["S", "W", "E"]), ("S", ["N", "E", "W"]), ("E", ["W", "N", "S"]), ("W", ["E", "S", "N"])]:
            flows += f'<flow id="f_{origin}{dests[0]}" type="mixed_traffic" route="{origin}{dests[0]}" begin="0" end="1000" probability="{prob * 0.5}"/>\n'
            flows += f'<flow id="f_{origin}{dests[1]}" type="mixed_traffic" route="{origin}{dests[1]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'
            flows += f'<flow id="f_{origin}{dests[2]}" type="mixed_traffic" route="{origin}{dests[2]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'

    with open(output_path, "w") as f:
        f.write(header + flows + "</routes>")

    return total_vehicles
