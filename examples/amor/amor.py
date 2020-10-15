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
    assume beam, and thus z axis and y=0, is half way along height of blade
    """
    half_strips_per_blade = strips_per_blade * 0.5
    return np.linspace(
        -half_strips_per_blade * strip_pitch_m,
        half_strips_per_blade * strip_pitch_m,
        strips_per_blade,
    )


def midpoint_between_wires_radial_direction() -> np.ndarray:
    half_wire_pitch_m = wire_pitch_m * 0.5
    return np.linspace(
        -half_wire_pitch_m, wires_per_blade * wire_pitch_m, wires_per_blade
    )


def rotate_around_y(angle_degrees: float, vertex: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return rotation_matrix.dot(vertex)


def create_winding_order() -> np.ndarray:
    vertices_per_pixel = 4
    winding_order = np.zeros((vertices_per_pixel, wires_per_blade * strips_per_blade))
    for strip_number in range(strips_per_blade):
        for wire_number in range(wires_per_blade):
            pixel_number = wire_number + strip_number * wires_per_blade
            winding_order[0][pixel_number] = (
                strip_number * (wires_per_blade + 1) + wire_number
            )
            winding_order[1][pixel_number] = (strip_number + 1) * (
                wires_per_blade + 1
            ) + wire_number
            winding_order[2][pixel_number] = (
                (strip_number + 1) * (wires_per_blade + 1) + wire_number + 1
            )
            winding_order[3][pixel_number] = (
                strip_number * (wires_per_blade + 1) + wire_number + 1
            )
    return winding_order


def construct_blade(blade_number: int):
    # The detector pixels are squares on a plane that corresponds to the front surface of the substrate

    # Create vertices for pixel corners as if the blade was in the YZ plane
    y = get_height_of_edges_of_each_strip()
    z = midpoint_between_wires_radial_direction()
    yy, zz = np.meshgrid(y, z)
    xx = np.zeros_like(yy)
    vertices = np.stack((xx, yy, zz))
    # reshape to a flat list of vertices
    vertices = np.reshape(vertices, (3, wires_per_blade * strips_per_blade))

    winding_order = create_winding_order() + blade_number * vertices.shape[1]

    # Rotate the blade around y
    rotation_angle_deg = 5 + blade_number * angle_between_blades_deg
    # TODO Vectorize rotate_around_y?


if __name__ == "__main__":
    construct_blade(0)
