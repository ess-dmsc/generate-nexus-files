import numpy as np
import pandas as pd
from tqdm import trange
from nexusutils.nexusbuilder import NexusBuilder
from nexusjson.nexus_to_json import NexusToDictConverter, object_to_json_file
import nexusformat.nexus as nexus
from contextlib import contextmanager
from typing import Dict
import h5py

"""
Generates example file with geometry for AMOR instrument with multiblade detector

z axis is horizontal and along neutron beam
x axis is the other horizontal axis
y axis is vertical
"""

# Parameters for detector blades
WIRE_PITCH_m = 0.004
STRIP_PITCH_m = 0.004
ANGLE_BETWEEN_BLADES_deg = 0.1448  # lowercase delta on diagrams
SAMPLE_TO_CLOSEST_SUBSTRATE_EDGE_m = 4.0  # R on diagrams
WIRES_PER_BLADE = 32
STRIPS_PER_BLADE = 32
ANGLE_BETWEEN_SUBSTRATE_AND_NEUTRON_deg = 5.0  # theta on diagrams
NUMBER_OF_BLADES = 9  # Maybe only 6 currently with digitisers?

INSTRUMENT_NAME = "AMOR"
EVENT_TOPIC = "FREIA_detector"
EVENT_SOURCE_NAME = "AMOR_EFU"
FORWARDER_TOPIC = "AMOR_forwarderData"
# NICOS_CACHE_TOPIC = "AMOR_nicosHistoryCache"


def get_edges_of_each_strip() -> np.ndarray:
    """
    These are the strip edge values for the pixel corner vertices
    assume beam is centred on the detector in this dimension and thus origin is in the middle
    """
    half_strips_per_blade = STRIPS_PER_BLADE * 0.5
    return np.linspace(
        -half_strips_per_blade * STRIP_PITCH_m,
        half_strips_per_blade * STRIP_PITCH_m,
        STRIPS_PER_BLADE + 1,
    )


def midpoint_between_wires_radial_direction() -> np.ndarray:
    return np.linspace(WIRES_PER_BLADE * WIRE_PITCH_m, 0.0, WIRES_PER_BLADE + 1)


def rotate_around_x(angle_degrees: float, vertex: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return rotation_matrix.dot(vertex)


def create_winding_order() -> np.ndarray:
    vertices_per_pixel = 4
    winding_order = np.zeros(
        (WIRES_PER_BLADE * STRIPS_PER_BLADE, vertices_per_pixel), dtype=np.int32
    )
    for strip_number in range(STRIPS_PER_BLADE):
        for wire_number in range(WIRES_PER_BLADE):
            pixel_number = wire_number + strip_number * WIRES_PER_BLADE
            winding_order[pixel_number][0] = (
                strip_number * (WIRES_PER_BLADE + 1) + wire_number
            )
            winding_order[pixel_number][1] = (strip_number + 1) * (
                WIRES_PER_BLADE + 1
            ) + wire_number
            winding_order[pixel_number][2] = (
                (strip_number + 1) * (WIRES_PER_BLADE + 1) + wire_number + 1
            )
            winding_order[pixel_number][3] = (
                strip_number * (WIRES_PER_BLADE + 1) + wire_number + 1
            )
    return winding_order


def write_to_off_file(
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
):
    """
    Write mesh geometry to a file in the OFF format
    https://en.wikipedia.org/wiki/OFF_(file_format)

    Viewable in geomview, meshlab etc.
    """
    number_of_vertices = vertices.shape[0]
    number_of_faces = faces.shape[0]
    vertices_per_face = faces.shape[1]
    with open(filename, "w") as f:
        f.writelines(
            (
                "OFF\n",
                f"# {INSTRUMENT_NAME}\n",
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
    x = get_edges_of_each_strip()
    z = midpoint_between_wires_radial_direction()
    xx, zz = np.meshgrid(x, z)
    yy = np.zeros_like(xx)
    vertices = np.stack((xx, yy, zz))
    # reshape to a flat list of vertices
    vertices = np.reshape(vertices, (3, (WIRES_PER_BLADE + 1) * (STRIPS_PER_BLADE + 1)))
    vertices = vertices.T

    winding_order = create_winding_order() + blade_number * vertices.shape[0]

    pixels_per_blade = WIRES_PER_BLADE * STRIPS_PER_BLADE
    start_face_index = pixels_per_blade * blade_number
    end_face_index = pixels_per_blade * (blade_number + 1)
    start_pixel_id = start_face_index + 1  # Pixel IDs are 1-indexed
    end_pixel_id = end_face_index + 1
    pixel_ids = np.column_stack(
        (
            np.arange(start_face_index, end_face_index),
            np.arange(start_pixel_id, end_pixel_id),
        )
    )

    transformed_vertices = np.zeros_like(vertices)
    # This ensures we create the blades in the order that matches the detector IDs output by the EFU
    blade_index = abs(blade_number - NUMBER_OF_BLADES) - 1
    for index, vertex in enumerate(vertices):
        vertex = rotate_around_x(-5.0, vertex)
        # Translation from sample position so we can rotate the blade a small angle on a wide arc
        vertex[2] += SAMPLE_TO_CLOSEST_SUBSTRATE_EDGE_m
        transformed_vertices[index, :] = rotate_around_x(
            -ANGLE_BETWEEN_BLADES_deg * blade_index, vertex
        )
        # Shift blade back so that its front edge is at z=0 again
        vertex[2] -= SAMPLE_TO_CLOSEST_SUBSTRATE_EDGE_m

    return transformed_vertices, winding_order, pixel_ids


def __add_attributes_to_group(group: h5py.Group, attributes: Dict):
    for key in attributes:
        if isinstance(attributes[key], str):
            # Since python 3 we have to treat strings like this
            group.attrs.create(
                key,
                np.array(attributes[key]).astype("|S" + str(len(attributes[key]))),
            )
        else:
            group.attrs.create(key, np.array(attributes[key]))


def write_to_nexus_file(
    filename: str,
    vertices: np.ndarray,
    voxels: np.ndarray,
    detector_ids: np.ndarray,
):
    winding_order = voxels.flatten().astype(np.int32)

    vertices_in_face = 4
    faces = np.arange(0, winding_order.size, vertices_in_face)

    with NexusBuilder(
        filename, compress_type="gzip", compress_opts=1, nx_entry_name="entry"
    ) as builder:
        instrument_group = builder.add_instrument(INSTRUMENT_NAME)
        detector_group = builder.add_nx_group(
            instrument_group, "multiblade_detector", "NXdetector"
        )
        shape_group = builder.add_nx_group(
            detector_group, "detector_shape", "NXoff_geometry"
        )
        builder.add_dataset(shape_group, "vertices", vertices.astype(np.float64))
        builder.add_dataset(shape_group, "winding_order", winding_order)
        builder.add_dataset(shape_group, "faces", faces.astype(np.int32))
        builder.add_dataset(
            shape_group, "detector_faces", detector_ids.astype(np.int32)
        )

        builder.add_dataset(
            detector_group,
            "detector_number",
            np.unique(detector_ids[:, 1]).astype(np.int32),
        )

        transforms_group = builder.add_nx_group(
            detector_group, "transformations", "NXtransformations"
        )
        builder.add_transformation(
            transforms_group,
            "translation",
            -4.1,
            "m",
            [0.0, 0.0, 1.0],
            name="translation",
        )
        detector_height = builder.add_nx_group(
            transforms_group,
            "COZ",
            "NXlog",
        )
        __add_attributes_to_group(
            detector_height,
            {
                "depends_on": "/entry/instrument/multiblade_detector/transformations/translation",
                "transformation_type": "translation",
                "units": "m",
                "vector": [0.0, -1.0, 0.0],
            },
        )
        detector_orientation = builder.add_nx_group(transforms_group, "COM", "NXlog")
        __add_attributes_to_group(
            detector_orientation,
            {
                "depends_on": "/entry/instrument/multiblade_detector/transformations/COZ",
                "transformation_type": "rotation",
                "units": "deg",
                "vector": [1.0, 0.0, 0.0],
            },
        )
        detector_pivot_point = builder.add_transformation(
            transforms_group,
            "translation",
            0.1,
            "m",
            [0.0, 0.0, 1.0],
            name="detector_pivot_point",
            depends_on=detector_orientation,
        )

        builder.add_depends_on(detector_group, detector_pivot_point)

        sample_group = builder.add_sample()
        sample_transforms_group = builder.add_nx_group(
            sample_group, "transformations", "NXtransformations"
        )
        sample_height = builder.add_nx_group(
            sample_transforms_group,
            "SOZ",
            "NXlog",
        )
        __add_attributes_to_group(
            sample_height,
            {
                "depends_on": ".",
                "transformation_type": "translation",
                "units": "m",
                "vector": [0.0, -1.0, 0.0],
            },
        )
        sample_orientation = builder.add_nx_group(
            sample_transforms_group,
            "SOM",
            "NXlog",
        )
        __add_attributes_to_group(
            sample_orientation,
            {
                "depends_on": "/entry/sample/transformations/SOZ",
                "transformation_type": "translation",
                "units": "deg",
                "vector": [1.0, 0.0, 0.0],
            },
        )
        builder.add_depends_on(sample_group, sample_orientation)

        builder.add_source("virtual_source", position=[0.0, 0.0, 30.0])

        builder.add_fake_event_data(1, 100)
        # Remove link to event data in the NXentry
        del builder.root["event_data_multiblade_detector"]


def __add_data_stream(streams, topic, source, path, module, value_type=None):
    options = {"topic": topic, "source": source, "writer_module": module}
    if value_type is not None:
        options["type"] = value_type

    streams[path] = options


@contextmanager
def nexus_file(nexus_filename: str):
    open_file = nexus.nxload(nexus_filename)
    try:
        yield open_file
    finally:
        open_file.close()


def write_to_json_file(nexus_filename: str, json_filename: str):
    converter = NexusToDictConverter()

    with nexus_file(nexus_filename) as nxs_file:
        streams = {}
        __add_data_stream(
            streams,
            EVENT_TOPIC,
            EVENT_SOURCE_NAME,
            "/entry/instrument/multiblade_detector/event_data",
            "ev42",
        )
        __add_data_stream(
            streams,
            FORWARDER_TOPIC,
            "COM",
            "/entry/instrument/multiblade_detector/transformations/COM",
            "f142",
        )
        __add_data_stream(
            streams,
            FORWARDER_TOPIC,
            "COZ",
            "/entry/instrument/multiblade_detector/transformations/COZ",
            "f142",
        )
        __add_data_stream(
            streams,
            FORWARDER_TOPIC,
            "SOM",
            "/entry/sample/transformations/SOM",
            "f142",
        )
        __add_data_stream(
            streams,
            FORWARDER_TOPIC,
            "SOZ",
            "/entry/sample/transformations/SOZ",
            "f142",
        )
        links = {}
        nexus_structure = converter.convert(nxs_file, streams, links)
        object_to_json_file(nexus_structure, json_filename)


if __name__ == "__main__":
    total_vertices = None
    total_faces = None
    total_ids = None
    for blade_number in trange(NUMBER_OF_BLADES):
        vertices, faces, detector_ids = construct_blade(blade_number)
        if total_vertices is None:
            total_vertices = vertices
            total_faces = faces
            total_ids = detector_ids
        else:
            total_vertices = np.vstack((total_vertices, vertices))
            total_faces = np.vstack((total_faces, faces))
            total_ids = np.vstack((total_ids, detector_ids))

    write_to_off_file(f"{INSTRUMENT_NAME}_multiblade.off", total_vertices, total_faces)

    nexus_filename = f"{INSTRUMENT_NAME}_multiblade.nxs"
    write_to_nexus_file(
        nexus_filename,
        total_vertices,
        total_faces,
        total_ids,
    )

    write_to_json_file(nexus_filename, "AMOR_nexus_structure.json")
