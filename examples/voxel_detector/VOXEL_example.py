from nexusutils.nexusbuilder import NexusBuilder
import numpy as np
import datetime

"""
Small example with detector described by an NXoff_geometry group where
each pixel is a volume defined by multiple faces in the mesh

Created to test loading such a geometry in Mantid 
"""


def add_voxel_detector(nexus_builder: NexusBuilder):
    detector_group = nexus_builder.add_detector_minimal("voxel geometry detector", 1)
    # Shape is a regular octahedron
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -2.0],
            [0.0, 1.0, -2.0],
            [-1.0, 0.0, -2.0],
            [0.0, -1.0, -2.0],
            [0.0, 0.0, -3.0],
        ]
    )
    # Number of vertices followed by vertex indices for each face
    off_faces = np.array(
        [
            [3, 1, 0, 4],
            [3, 4, 0, 3],
            [3, 3, 0, 2],
            [3, 2, 0, 1],
            [3, 1, 5, 2],
            [3, 2, 5, 3],
            [3, 3, 5, 4],
            [3, 4, 5, 1],
            [3, 6, 5, 9],
            [3, 9, 5, 8],
            [3, 8, 5, 7],
            [3, 7, 5, 6],
            [3, 6, 10, 7],
            [3, 7, 10, 8],
            [3, 8, 10, 9],
            [3, 9, 10, 6],
        ]
    )
    detector_numbers = np.array([8, 9])
    # Map all faces to the same detector number such that the whole shape is defined as a single voxel
    detector_faces = np.array(
        [
            [0, detector_numbers[0]],
            [1, detector_numbers[0]],
            [2, detector_numbers[0]],
            [3, detector_numbers[0]],
            [4, detector_numbers[0]],
            [5, detector_numbers[0]],
            [6, detector_numbers[0]],
            [7, detector_numbers[0]],
            [8, detector_numbers[1]],
            [9, detector_numbers[1]],
            [10, detector_numbers[1]],
            [11, detector_numbers[1]],
            [12, detector_numbers[1]],
            [13, detector_numbers[1]],
            [14, detector_numbers[1]],
            [15, detector_numbers[1]],
        ]
    )
    nexus_builder.add_shape(
        detector_group, "detector_shape", vertices, off_faces, detector_faces
    )
    nexus_builder.add_dataset(detector_group, "detector_number", detector_numbers)

    transform_group = nexus_builder.add_nx_group(
        detector_group, "transformations", "NXtransformation"
    )
    position = nexus_builder.add_transformation(
        transform_group,
        "translation",
        np.array([2.0]),
        units="m",
        vector=[1.0, 0.0, 1.0],
        name="position",
    )
    nexus_builder.add_dataset(detector_group, "depends_on", position.name)

    # Record the voxel position
    nexus_builder.add_dataset(detector_group, "x_pixel_offset", [1.1, 1.1])
    nexus_builder.add_dataset(detector_group, "y_pixel_offset", [2.2, 2.2])
    nexus_builder.add_dataset(detector_group, "z_pixel_offset", [3.3, 1.3])


if __name__ == "__main__":
    output_filename = "VOXEL_example.nxs"
    with NexusBuilder(
        output_filename,
        compress_type="gzip",
        compress_opts=1,
        nx_entry_name="entry",
    ) as builder:
        builder.add_instrument("VOXEL", "instrument")
        sample_group = builder.add_sample("test sample")
        builder.add_dataset(sample_group, "name", "test_sample")
        source_group = builder.add_source("source")
        transforms_group = builder.add_nx_group(
            source_group, "transformations", "NXtransformation"
        )
        source_position = builder.add_transformation(
            transforms_group,
            "translation",
            np.array([20.0]),
            units="m",
            vector=[0.0, 0.0, -1.0],
            name="position",
        )
        builder.add_dataset(source_group, "depends_on", source_position.name)
        builder.add_dataset(builder.root, "name", "VOXEL", {"short_name": "VOXEL"})

        add_voxel_detector(builder)

        # Add some event data and a start_time dataset so we can load with Mantid
        builder.add_fake_event_data(1, 100)
        builder.get_root()["start_time"] = datetime.datetime.now().isoformat()
