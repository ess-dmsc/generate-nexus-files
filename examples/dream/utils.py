import numpy as np
from nexusutils.nexusbuilder import NexusBuilder


def write_to_nexus_file(
    filename: str,
    vertices: np.ndarray,
    voxels: np.ndarray,
    detector_ids: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    z_offsets: np.ndarray,
):
    # Slice off first column of voxels as it contains number of vertices in the face (from OFF format)
    # in NeXus that information is carried by the "faces" dataset
    winding_order = voxels[:, 1:].flatten().astype(np.int32)

    vertices_in_face = 4
    faces = np.arange(0, winding_order.size, vertices_in_face)

    with NexusBuilder(
        filename, compress_type="gzip", compress_opts=1, nx_entry_name="entry"
    ) as builder:
        instrument_group = builder.add_nx_group(builder.root, "DREAM", "NXinstrument")
        builder.instrument = instrument_group
        builder.add_dataset(instrument_group, "name", "DREAM")
        detector_group = builder.add_nx_group(
            instrument_group, "endcap_detector", "NXdetector"
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

        # Record voxel centre positions
        builder.add_dataset(detector_group, "x_pixel_offset", x_offsets, {"units": "m"})
        builder.add_dataset(detector_group, "y_pixel_offset", y_offsets, {"units": "m"})
        builder.add_dataset(detector_group, "z_pixel_offset", z_offsets, {"units": "m"})

        # Add a source, sample, start_time and some events so that we can easily load the file into Mantid
        sample_group = builder.add_sample("test sample")
        builder.add_dataset(sample_group, "name", "test_sample")
        source_group = builder.add_source("source")
        transforms_group = builder.add_nx_group(
            source_group, "transformations", "NXtransformation"
        )
        builder.add_transformation(
            transforms_group,
            "translation",
            np.array([20.0]),
            units="m",
            vector=[0.0, 0.0, -1.0],
            name="position",
        )
        builder.add_fake_event_data(1, 100)
        builder.get_root()["start_time"] = datetime.datetime.now().isoformat()


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
                "# DREAM End-Cap\n",
                f"{number_of_vertices} {number_of_faces} 0\n",
            )
        )
    with open(filename, "a") as f:
        pd.DataFrame(vertices).to_csv(f, sep=" ", header=None, index=False)
    with open(filename, "a") as f:
        pd.DataFrame(voxels).to_csv(f, sep=" ", header=None, index=False)
