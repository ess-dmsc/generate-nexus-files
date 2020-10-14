import numpy as np

"""
Generates example file with geometry for AMOR instrument with multiblade detector

z axis is horizontal and along neutron beam
x axis is the other horizontal axis
y axis is vertical
"""

# Parameters for detector blades
wire_pitch_m = 0.004
strip_pitch_m = 0.004
angle_between_blades_deg = 0.1448  # lowercase delta on diagrams
sample_to_closest_wire_m = 4.0  # R on diagrams
wires_per_blade = 32
strips_per_blade = 32
angle_between_substrate_and_neutron_deg = 5.0  # theta on diagrams


def get_height_of_edges_of_each_strip() -> np.ndarray:
    """
    These are the y values for the pixel corner vertices
    assume beam, and thus z axis, is half way along height of blade
    """
    half_strips_per_blade = strips_per_blade * 0.5
    return np.linspace(
        -half_strips_per_blade * strip_pitch_m,
        half_strips_per_blade * strip_pitch_m,
        strips_per_blade,
    )


if __name__ == "__main__":
    pass
