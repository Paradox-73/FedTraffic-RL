import random


def generate_routes(experiment_name, output_path):
    """
    Generates the hello.rou.xml file.
    This file tells SUMO:
    1. What kind of vehicles exist (Cars, Buses, Motos).
    2. Where the roads go (Routes).
    3. When vehicles appear (Flows).
    """

    # --- XML HEADER & VEHICLE DEFINITIONS ---
    # We use a "vTypeDistribution" to randomly assign vehicle types based on probability.
    # Total probability sums to 1.00 (0.80 + 0.10 + 0.10).
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

    # We start with an empty string and append flows based on the difficulty stage
    flows = ""

    # --- STAGE 1 LOGIC: The Baseline ---
    if "stage1" in experiment_name:
        # LOGIC: Uniform, constant traffic.
        # We add a random multiplier (0.9 to 1.1) so every episode is slightly different.
        base_prob = random.uniform(0.05, 0.1)

        # 80% of traffic goes Straight (NS, SN, EW, WE)
        # 20% of traffic Turns (NW, SE, EN, WS).
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
        # LOGIC: Non-Stationary traffic (Changes over time).
        # We pick a random time for the "Morning Rush" to start (between step 200 and 400).
        peak_start = random.randint(200, 400)
        peak_end = peak_start + 400

        # PHASE 1: THE LULL (Start -> Peak)
        # Very low traffic (prob=0.1) on all sides. Everyone is asleep.
        lull_prob = 0.05
        flows += f'<flow id="lull_NS" type="mixed_traffic" route="NS" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_SN" type="mixed_traffic" route="SN" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_EW" type="mixed_traffic" route="EW" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'
        flows += f'<flow id="lull_WE" type="mixed_traffic" route="WE" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'

        # PHASE 2: THE RUSH (Peak Start -> Peak End)
        # Traffic spikes massively!
        # ASYMMETRY: North-South is the "Main Artery" (0.6 prob). East-West is minor (0.25).
        # The agent must learn to prioritize NS green lights here.
        ns_rush_prob = 0.15
        ew_rush_prob = 0.0625
        flows += f'<flow id="rush_NS" type="mixed_traffic" route="NS" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_SN" type="mixed_traffic" route="SN" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_EW" type="mixed_traffic" route="EW" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'
        flows += f'<flow id="rush_WE" type="mixed_traffic" route="WE" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'

        # PHASE 3: COOLDOWN (Peak End -> 1000)
        # Rush hour is over, traffic returns to a medium balance.
        cool_prob = 0.05
        flows += f'<flow id="cool_NS" type="mixed_traffic" route="NS" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_SN" type="mixed_traffic" route="SN" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_EW" type="mixed_traffic" route="EW" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'
        flows += f'<flow id="cool_WE" type="mixed_traffic" route="WE" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'

    # --- STAGE 3 LOGIC: Gridlock ---
    elif "stage3" in experiment_name:
        # LOGIC: High conflict turns.
        # This loop generates traffic from every direction (N, S, E, W).
        # origin="N", dests=["S", "W", "E"] means traffic coming from North goes to S, W, and E.
        prob = random.uniform(0.05, 0.1)
        for origin, dests in [("N", ["S", "W", "E"]), ("S", ["N", "E", "W"]), ("E", ["W", "N", "S"]), ("W", ["E", "S", "N"])]:
            # 50% go Straight (e.g., N -> S)
            flows += f'<flow id="f_{origin}{dests[0]}" type="mixed_traffic" route="{origin}{dests[0]}" begin="0" end="1000" probability="{prob * 0.5}"/>\n'
            # 25% Turn Right (e.g., N -> W)
            flows += f'<flow id="f_{origin}{dests[1]}" type="mixed_traffic" route="{origin}{dests[1]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'
            # 25% Turn Left (e.g., N -> E)
            # CRITICAL: Left turns block the intersection in SUMO (yielding logic). This creates gridlock.
            flows += f'<flow id="f_{origin}{dests[2]}" type="mixed_traffic" route="{origin}{dests[2]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'

    # --- FILE WRITING ---
    # Combine Header + Flows + Closing Tag and write to the .rou.xml file
    with open(output_path, "w") as f:
        f.write(header + flows + "</routes>")
