import numpy as np
import pandas as pd
from tqdm import trange

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
        strips_per_blade + 1,
    )


def midpoint_between_wires_radial_direction() -> np.ndarray:
    half_wire_pitch_m = wire_pitch_m * 0.5
    return np.linspace(
        -half_wire_pitch_m, wires_per_blade * wire_pitch_m, wires_per_blade + 1
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
    winding_order = np.zeros(
        (wires_per_blade * strips_per_blade, vertices_per_pixel), dtype=np.int32
    )
    for strip_number in range(strips_per_blade):
        for wire_number in range(wires_per_blade):
            pixel_number = wire_number + strip_number * wires_per_blade
            winding_order[pixel_number][0] = (
                strip_number * (wires_per_blade + 1) + wire_number
            )
            winding_order[pixel_number][1] = (strip_number + 1) * (
                wires_per_blade + 1
            ) + wire_number
            winding_order[pixel_number][2] = (
                (strip_number + 1) * (wires_per_blade + 1) + wire_number + 1
            )
            winding_order[pixel_number][3] = (
                strip_number * (wires_per_blade + 1) + wire_number + 1
            )
    return winding_order


def write_to_off_file(
    filename: str, vertices: np.ndarray, faces: np.ndarray,
):
    """
    Write mesh geometry to a file in the OFF format
    https://en.wikipedia.org/wiki/OFF_(file_format)
    """
    number_of_vertices = vertices.shape[0]
    number_of_faces = faces.shape[0]
    vertices_per_face = faces.shape[1]
    with open(filename, "w") as f:
        f.writelines(
            (
                "OFF\n",
                "# AMOR Multiblade\n",
                f"{number_of_vertices} {number_of_faces} 0\n",
            )
        )
    with open(filename, "a") as f:
        pd.DataFrame(vertices).to_csv(f, sep=" ", header=None, index=False)

    # Prepend winding order with number of vertices in face
    off_faces = np.hstack(
        (vertices_per_face * np.ones((number_of_faces, 1), dtype=np.int32), faces)
    )
    with open(filename, "a") as f:
        pd.DataFrame(off_faces).to_csv(f, sep=" ", header=None, index=False)


def construct_blade(blade_number: int) -> (np.ndarray, np.ndarray, np.ndarray):
    # The detector pixels are squares on a plane that corresponds to the front surface of the substrate

    # Create vertices for pixel corners as if the blade was in the YZ plane
    y = get_height_of_edges_of_each_strip()
    z = midpoint_between_wires_radial_direction()
    yy, zz = np.meshgrid(y, z)
    xx = np.zeros_like(yy)
    vertices = np.stack((xx, yy, zz))
    # reshape to a flat list of vertices
    vertices = np.reshape(vertices, (3, (wires_per_blade + 1) * (strips_per_blade + 1)))
    vertices = vertices.T

    winding_order = create_winding_order() + blade_number * vertices.shape[0]

    pixels_per_blade = wires_per_blade * strips_per_blade
    start_face_index = pixels_per_blade * blade_number
    end_face_index = pixels_per_blade * (blade_number + 1)
    start_pixel_id = start_face_index + 1  # Pixel IDs are 1-indexed
    end_pixel_id = end_face_index + 1
    pixel_ids = np.column_stack(
        np.arange(start_face_index, end_face_index),
        np.arange(start_pixel_id, end_pixel_id),
    )

    transformed_vertices = np.zeros_like(vertices)
    for index, vertex in enumerate(vertices):
        # Rotate the blade around y
        vertex = rotate_around_y(5.0, vertex)
        # Translation from sample position
        vertex[2] += sample_to_closest_wire_m
        transformed_vertices[index, :] = rotate_around_y(
            angle_between_blades_deg * blade_number, vertex
        )

    return transformed_vertices, winding_order, pixel_ids


if __name__ == "__main__":
    total_vertices = None
    total_faces = None
    total_ids = None
    for blade_number in trange(9):
        vertices, faces, detector_ids = construct_blade(blade_number)
        if total_vertices is None:
            total_vertices = vertices
            total_faces = faces
            total_ids = detector_ids
        else:
            total_vertices = np.vstack((total_vertices, vertices))
            total_faces = np.vstack((total_faces, faces))
            total_ids = np.vstack((total_ids, detector_ids))

    write_to_off_file("amor.off", total_vertices, total_faces)
