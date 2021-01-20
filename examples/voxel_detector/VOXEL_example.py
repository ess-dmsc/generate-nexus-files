from nexusutils.nexusbuilder import NexusBuilder
import numpy as np
import datetime
import pandas as pd

"""
Small example with detector described by an NXoff_geometry group where
each pixel is a volume defined by multiple faces in the mesh

Created to test loading such a geometry in Mantid 
"""


def add_voxel_detector(nexus_builder: NexusBuilder, n_voxels: int = 3):
    detector_group = nexus_builder.add_detector_minimal("voxel geometry detector", 1)

    vertices = np.array([[0.0, 0.0, 0.0]])
    off_faces = np.empty((0, 4), dtype=int)
    detector_numbers = np.arange(n_voxels)
    detector_faces = np.empty((0, 2), dtype=int)
    for voxel_number in range(n_voxels):
        # Each voxel is a regular octahedron
        new_vertices = np.array(
            [
                [0.0, 0.0, 1.0 - 2 * voxel_number],
                [1.0, 0.0, 0.0 - 2 * voxel_number],
                [0.0, 1.0, 0.0 - 2 * voxel_number],
                [-1.0, 0.0, 0.0 - 2 * voxel_number],
                [0.0, -1.0, 0.0 - 2 * voxel_number],
                [0.0, 0.0, -1.0 - 2 * voxel_number],
            ]
        )
        vertices = np.append(vertices[:-1, :], new_vertices, axis=0)

        # Number of vertices followed by vertex indices for each face
        # the first column doesn't end up in the NeXus file dataset
        new_off_faces = np.array(
            [
                [3, 1 + 5 * voxel_number, 5 * voxel_number, 4 + 5 * voxel_number],
                [3, 4 + 5 * voxel_number, 5 * voxel_number, 3 + 5 * voxel_number],
                [3, 3 + 5 * voxel_number, 5 * voxel_number, 2 + 5 * voxel_number],
                [3, 2 + 5 * voxel_number, 5 * voxel_number, 1 + 5 * voxel_number],
                [3, 1 + 5 * voxel_number, 5 + 5 * voxel_number, 2 + 5 * voxel_number],
                [3, 2 + 5 * voxel_number, 5 + 5 * voxel_number, 3 + 5 * voxel_number],
                [3, 3 + 5 * voxel_number, 5 + 5 * voxel_number, 4 + 5 * voxel_number],
                [3, 4 + 5 * voxel_number, 5 + 5 * voxel_number, 1 + 5 * voxel_number],
            ]
        )
        off_faces = np.append(off_faces, new_off_faces, axis=0)

        detector_number = detector_numbers[voxel_number]
        # Map 8 faces to each detector number
        new_detector_faces = np.array(
            [
                [detector_number * 8, detector_number],
                [1 + detector_number * 8, detector_number],
                [2 + detector_number * 8, detector_number],
                [3 + detector_number * 8, detector_number],
                [4 + detector_number * 8, detector_number],
                [5 + detector_number * 8, detector_number],
                [6 + detector_number * 8, detector_number],
                [7 + detector_number * 8, detector_number],
            ]
        )
        detector_faces = np.append(detector_faces, new_detector_faces, axis=0)

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

    # Record the voxel positions
    x_offsets = 1.1 * np.ones(n_voxels, dtype=float)
    y_offsets = 2.2 * np.ones(n_voxels, dtype=float)
    z_offsets = np.arange(n_voxels, -n_voxels, 2.0, dtype=float)
    nexus_builder.add_dataset(detector_group, "x_pixel_offset", x_offsets)
    nexus_builder.add_dataset(detector_group, "y_pixel_offset", y_offsets)
    nexus_builder.add_dataset(detector_group, "z_pixel_offset", z_offsets)

    write_to_off_file(
        "voxel.off", vertices.shape[0], off_faces.shape[0], vertices, off_faces
    )


def write_to_off_file(
    filename: str,
    number_of_vertices: int,
    number_of_faces: int,
    vertices: np.ndarray,
    voxels: np.ndarray,
):
    """
    Write mesh geometry to a file in the OFF format
    https://en.wikipedia.org/wiki/OFF_(file_format)
    """
    with open(filename, "w") as f:
        f.writelines(
            (
                "OFF\n",
                "# Example VOXEL detector\n",
                f"{number_of_vertices} {number_of_faces} 0\n",
            )
        )
    with open(filename, "a") as f:
        pd.DataFrame(vertices).to_csv(f, sep=" ", header=None, index=False)
    with open(filename, "a") as f:
        pd.DataFrame(voxels).to_csv(f, sep=" ", header=None, index=False)


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
