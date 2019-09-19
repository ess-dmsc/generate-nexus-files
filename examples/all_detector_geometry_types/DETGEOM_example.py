from nexusutils.nexusbuilder import NexusBuilder
import numpy as np


def add_per_pixel_mesh_geometry_detector():
    detector_group = builder.add_detector_minimal("per pixel mesh geometry detector", 1)
    vertices = np.array(
        [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0]])
    # Number of vertices followed by vertex indices for each face
    off_faces = np.array([[4, 3, 2, 1, 0]])
    builder.add_shape(detector_group, "pixel_shape", vertices, off_faces)

    x_offsets = np.linspace(start=-0.5, stop=0.5, num=2)
    y_offsets = np.linspace(start=-0.5, stop=0.5, num=2)
    x_pixel_offsets, y_pixel_offsets = np.meshgrid(x_offsets, y_offsets)
    detector_numbers = np.array([0, 1, 2, 3])
    builder.add_dataset(detector_group, "detector_number", detector_numbers)
    builder.add_dataset(detector_group, "x_pixel_offset", x_pixel_offsets.flatten())
    builder.add_dataset(detector_group, "y_pixel_offset", y_pixel_offsets.flatten())

    transform_group = builder.add_nx_group(detector_group, 'transformations', 'NXtransformation')
    position = builder.add_transformation(transform_group, "translation", np.array([1.0]), units="m",
                                          vector=[-1.0, 0.0, 0.0], name="position")
    builder.add_dataset(detector_group, "depends_on", position.name)


def add_per_pixel_cylinder_geometry_detector():
    detector_group = builder.add_detector_minimal("per pixel cylinder geometry detector", 2)
    builder.add_tube_pixel(detector_group, height=0.5, radius=0.25,
                           axis=np.array([0.0, 0.0, 1.0]), centre=np.array([0.0, 0.0, 0.25]))

    x_offsets = np.linspace(start=-0.5, stop=0.5, num=2)
    y_offsets = np.linspace(start=-0.5, stop=0.5, num=2)
    x_pixel_offsets, y_pixel_offsets = np.meshgrid(x_offsets, y_offsets)
    detector_numbers = np.array([0, 1, 2, 3])
    builder.add_dataset(detector_group, "detector_number", detector_numbers)
    builder.add_dataset(detector_group, "x_pixel_offset", x_pixel_offsets.flatten())
    builder.add_dataset(detector_group, "y_pixel_offset", y_pixel_offsets.flatten())

    transform_group = builder.add_nx_group(detector_group, 'transformations', 'NXtransformation')
    position = builder.add_transformation(transform_group, "translation", np.array([1.0]), units="m",
                                          vector=[-1.0, 0.0, 1.0], name="position")
    builder.add_dataset(detector_group, "depends_on", position.name)


def add_complete_mesh_geometry_detector():
    detector_group = builder.add_detector_minimal("complete mesh geometry detector", 3)
    vertices = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # Number of vertices followed by vertex indices for each face
    off_faces = np.array(
        [[3, 1, 0, 4], [3, 4, 0, 3], [3, 3, 0, 2], [3, 2, 0, 1], [3, 1, 5, 2], [3, 2, 5, 3], [3, 3, 5, 4],
         [3, 4, 5, 1]])
    detector_faces = np.array([[4, 0], [5, 1], [6, 2], [7, 3]])
    builder.add_shape(detector_group, "detector_shape", vertices, off_faces, detector_faces)
    detector_numbers = np.array([0, 1, 2, 3])
    builder.add_dataset(detector_group, "detector_number", detector_numbers)

    transform_group = builder.add_nx_group(detector_group, 'transformations', 'NXtransformation')
    position = builder.add_transformation(transform_group, "translation", np.array([1.0]), units="m",
                                          vector=[1.0, 0.0, 1.0], name="position")
    builder.add_dataset(detector_group, "depends_on", position.name)


def add_complete_cylinder_geometry_detector():
    detector_group = builder.add_detector_minimal("complete cylinder geometry detector", 4)

    cylinder_group = builder.add_nx_group(detector_group, "detector_shape", "NXcylindrical_geometry")
    vertices = np.array([[0.0, -0.4, 0.0], [0.5, -0.4, 0.0], [0.0, 0.0, 0.0],
                         [0.5, 0.0, 0.0], [0.0, 0.3, 0.0],
                         [0.5, 0.3, 0.0], [0.0, 0.5, 0.0]])
    cylinders = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    cylinder_detector_number = np.array([[0, 0], [1, 1], [2, 2]])
    builder.add_dataset(cylinder_group, "vertices", vertices)
    builder.add_dataset(cylinder_group, "cylinders", cylinders)
    builder.add_dataset(cylinder_group, "detector_number", cylinder_detector_number)

    detector_numbers = np.array([0, 1, 2])
    builder.add_dataset(detector_group, "detector_number", detector_numbers)

    transform_group = builder.add_nx_group(detector_group, 'transformations', 'NXtransformation')
    position = builder.add_transformation(transform_group, "translation", np.array([1.0]), units="m",
                                          vector=[1.0, 0.0, 0.0], name="position")
    builder.add_dataset(detector_group, "depends_on", position.name)


if __name__ == '__main__':
    output_filename = 'DETGEOM_example.hdf5'
    with NexusBuilder(output_filename, compress_type='gzip', compress_opts=1) as builder:
        builder.add_instrument("DETGEOM", "instrument")
        sample_group = builder.add_sample("test sample")
        builder.add_dataset(sample_group, "name", "test_sample")
        source_group = builder.add_source("source")
        transform_group = builder.add_nx_group(source_group, "transformations", "NXtransformation")
        position = builder.add_transformation(transform_group, "translation", np.array([20.0]), units="m",
                                              vector=[0.0, 0.0, -1.0], name="position")
        builder.add_dataset(source_group, "depends_on", position.name)
        builder.add_dataset(builder.root, 'name', 'DETGEOM', {'short_name': 'DETGEOM'})

        add_per_pixel_mesh_geometry_detector()
        add_per_pixel_cylinder_geometry_detector()
        add_complete_mesh_geometry_detector()
        add_complete_cylinder_geometry_detector()
