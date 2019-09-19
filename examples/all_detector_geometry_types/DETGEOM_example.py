from nexusutils.nexusbuilder import NexusBuilder
from nexusutils.detectorplotter import DetectorPlotter
import numpy as np


def add_per_pixel_mesh_geometry_detector():
    detector_group = builder.add_detector_minimal("per pixel mesh geometry detector", 1)
    builder.add_shape(detector_group, "pixel_shape", )


def add_per_pixel_cylinder_geometry_detector():
    detector_group = builder.add_detector_minimal("per pixel cylinder geometry detector", 2)
    builder.add_shape(detector_group, "pixel_shape")


def add_complete_mesh_geometry_detector():
    detector_group = builder.add_detector_minimal("complete mesh geometry detector", 3)
    vertices = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # Number of vertices followed by vertex indices for each face
    off_faces = np.array(
        [[3, 1, 0, 4], [3, 4, 0, 3], [3, 3, 0, 2], [3, 2, 0, 1], [3, 1, 5, 2], [3, 2, 5, 3], [3, 3, 5, 4],
         [3, 4, 5, 1]])
    detector_faces = np.array([[0, 0], [1, 1], [2, 2]])
    builder.add_shape(detector_group, "detector_shape", vertices, off_faces, detector_faces)


def add_complete_cylinder_geometry_detector():
    detector_group = builder.add_detector_minimal("complete cylinder geometry detector", 4)
    builder.add_shape(detector_group, "pixel_shape")


if __name__ == '__main__':
    output_filename = 'DETGEOM_example.hdf5'
    with NexusBuilder(output_filename, compress_type='gzip', compress_opts=1) as builder:
        builder.add_instrument("DETGEOM", "instrument")
        builder.add_dataset(builder.root, 'name', 'DETGEOM', {'short_name': 'DETGEOM'})

        add_per_pixel_mesh_geometry_detector()
        add_per_pixel_cylinder_geometry_detector()
        add_complete_mesh_geometry_detector()
        add_complete_cylinder_geometry_detector()

        # det_ids = builder.add_fake_event_data(10, 10)

    # with DetectorPlotter(output_filename) as plotter:
    #    plotter.plot_pixel_positions()
