import random


def generate_routes(experiment_name, output_path):

    header = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vTypeDistribution id="mixed_traffic">
        <vType id="car" length="4.9" accel="2.3" maxSpeed="16.7"
               color="yellow" probability="0.75" speedFactor="1.0" speedDev="0.1"/>

        <vType id="bus" length="12.0" accel="1.0" maxSpeed="12.0"
               color="red" probability="0.15" speedFactor="0.9"/>

        <vType id="moto" length="2.0" accel="3.0" maxSpeed="20.0"
               color="blue" probability="0.10" speedFactor="1.1"/>
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

    # =========================
    # STAGE 1 — HEAVIER BASELINE
    # =========================
    if "stage1" in experiment_name:
        vol_scale = random.uniform(0.95, 1.05)

        # Increased from 0.15 → 0.35
        base_prob = 0.35 * vol_scale

        # 70% straight, 30% turning
        straight = base_prob * 0.7
        turn = base_prob * 0.3

        for route in ["NS", "SN", "EW", "WE"]:
            flows += f'<flow id="f_{route}" type="mixed_traffic" route="{route}" begin="0" end="1000" probability="{straight}"/>\n'

        for route in ["NW", "SE", "EN", "WS"]:
            flows += f'<flow id="f_{route}" type="mixed_traffic" route="{route}" begin="0" end="1000" probability="{turn}"/>\n'

    # =========================
    # STAGE 2 — STRONGER RUSH
    # =========================
    elif "stage2" in experiment_name:
        peak_start = random.randint(200, 350)
        peak_end = peak_start + 400

        # LULL
        lull_prob = 0.15
        for route in ["NS", "SN", "EW", "WE"]:
            flows += f'<flow id="lull_{route}" type="mixed_traffic" route="{route}" begin="0" end="{peak_start}" probability="{lull_prob}"/>\n'

        # RUSH
        ns_rush_prob = 0.75
        ew_rush_prob = 0.30

        flows += f'<flow id="rush_NS" type="mixed_traffic" route="NS" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_SN" type="mixed_traffic" route="SN" begin="{peak_start}" end="{peak_end}" probability="{ns_rush_prob}"/>\n'
        flows += f'<flow id="rush_EW" type="mixed_traffic" route="EW" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'
        flows += f'<flow id="rush_WE" type="mixed_traffic" route="WE" begin="{peak_start}" end="{peak_end}" probability="{ew_rush_prob}"/>\n'

        # COOLDOWN
        cool_prob = 0.25
        for route in ["NS", "SN", "EW", "WE"]:
            flows += f'<flow id="cool_{route}" type="mixed_traffic" route="{route}" begin="{peak_end}" end="1000" probability="{cool_prob}"/>\n'

    # =========================
    # STAGE 3 — REAL GRIDLOCK
    # =========================
    elif "stage3" in experiment_name:
        prob = 0.45  # increased from 0.3

        for origin, dests in [
            ("N", ["S", "W", "E"]),
            ("S", ["N", "E", "W"]),
            ("E", ["W", "N", "S"]),
            ("W", ["E", "S", "N"]),
        ]:
            flows += f'<flow id="f_{origin}{dests[0]}" type="mixed_traffic" route="{origin}{dests[0]}" begin="0" end="1000" probability="{prob * 0.5}"/>\n'
            flows += f'<flow id="f_{origin}{dests[1]}" type="mixed_traffic" route="{origin}{dests[1]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'
            flows += f'<flow id="f_{origin}{dests[2]}" type="mixed_traffic" route="{origin}{dests[2]}" begin="0" end="1000" probability="{prob * 0.25}"/>\n'

    with open(output_path, "w") as f:
        f.write(header + flows + "</routes>")