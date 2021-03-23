"""
Create a simple file from scratch, describing an instrument comprising a sample and rectangular detector
Script is self contained, requires only h5py and numpy
"""

import h5py
import numpy as np


def create_nx_group(parent_node, group_name, nx_class_name):
    """
    Create an HDF5 group with an NX_class attribute
    :param parent_node: Parent node of the new group
    :param group_name: Name of the new group
    :param nx_class_name: NX_class name of the new group
    :return: The newly created group
    """
    new_group = parent_node.create_group(group_name)
    new_group.attrs.create('NX_class', np.array(nx_class_name).astype('|S{}'.format(len(nx_class_name))))
    return new_group


def create_dataset(parent_node, dataset_name, data, attributes=None):
    """
    Add a dataset to a given group

    :param parent_node: Parent node of the new dataset
    :param dataset_name: Name of the new dataset
    :param data: Data to put in the dataset
    :param attributes: Optional dictionary of attributes to add to dataset
    :return: The newly created dataset
    """
    if isinstance(data, str):
        dataset = parent_node.create_dataset(dataset_name, data=np.array(data).astype('|S' + str(len(data))))
    else:
        dataset = parent_node.create_dataset(dataset_name, data=data)

    if attributes:
        for key in attributes:
            if isinstance(attributes[key], str):
                dataset.attrs.create(key, np.array(attributes[key]).astype('|S{}'.format(len(attributes[key]))))
            else:
                dataset.attrs.create(key, np.array(attributes[key]))
    return dataset


def add_detector(parent_node, group_name):
    pixels_per_axis = 100
    pixel_width = 0.002
    detector_side_length = 0.4
    half_detector_side_length = detector_side_length / 2.0
    half_pixel_width = pixel_width / 2.0
    single_axis_offsets = (pixel_width * np.arange(0, pixels_per_axis, 1,
                                                   dtype=np.float)) - half_detector_side_length + half_pixel_width
    detector_group = create_nx_group(parent_node, group_name, 'NXdetector')
    x_offsets, y_offsets = np.meshgrid(single_axis_offsets,
                                       single_axis_offsets)
    create_dataset(detector_group, 'x_pixel_offset', x_offsets, {'units': 'm'})
    create_dataset(detector_group, 'y_pixel_offset', y_offsets, {'units': 'm'})

    create_dataset(detector_group, 'local_name', 'Simple rectangular detector')

    add_pixel_shape(detector_group)

    pixel_ids = np.arange(0, pixels_per_axis ** 2, 1, dtype=int)
    pixel_ids = np.reshape(pixel_ids, (pixels_per_axis, pixels_per_axis))
    create_dataset(detector_group, 'detector_number', pixel_ids)

    add_detector_transforms(detector_group)

    return detector_group


def add_pixel_shape(detector_group):
    """
    Add an NXoff_geometry group which describes the shape of a pixel
    :param detector_group:
    """
    pixel_shape = create_nx_group(detector_group, 'pixel_shape', 'NXoff_geometry')
    pixel_verts = np.array([[-0.001, -0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0], [-0.001, 0.001, 0.0]],
                           dtype=np.float32)
    pixel_winding_order = np.array([0, 1, 2, 3], dtype=np.int32)
    pixel_faces = np.array([0], dtype=np.int32)
    create_dataset(pixel_shape, 'faces', pixel_faces)
    create_dataset(pixel_shape, 'vertices', pixel_verts, {'units': 'm'})
    create_dataset(pixel_shape, 'winding_order', pixel_winding_order)


def add_detector_transforms(detector_group):
    """
    Add the transformations describing the position and orientation of the detector
    :param detector_group:
    """
    transforms = create_nx_group(detector_group, 'transformations', 'NXtransformations')
    z_offset = create_dataset(transforms, 'beam_direction_offset', 0.5,
                              {'transformation_type': 'translation', 'units': 'm', 'vector': [0.0, 0.0, 1.0],
                               'depends_on': '.'})
    x_offset = create_dataset(transforms, 'horizontal_direction_offset', 0.15,
                              {'transformation_type': 'translation', 'units': 'm', 'vector': [1.0, 0.0, 0.0],
                               'depends_on': z_offset.name})
    orientation = create_dataset(transforms, 'orientation', np.pi / 4.0,
                                 {'transformation_type': 'rotation', 'units': 'rad', 'vector': [0.0, 1.0, 0.0],
                                  'depends_on': x_offset.name})
    create_dataset(detector_group, 'depends_on', orientation.name)


def add_source_position(source_group):
    transforms = create_nx_group(source_group, 'transformations', 'NXtransformations')
    create_dataset(transforms, 'beam_direction_offset', 5.0,
                   {'transformation_type': 'translation', 'units': 'm', 'vector': [0.0, 0.0, -1.0],
                    'depends_on': '.'})


if __name__ == '__main__':
    output_nexus_filename = 'simple_rectangular_detector.nxs'
    with h5py.File(output_nexus_filename, 'w') as output_file:
        entry = create_nx_group(output_file, 'entry', 'NXentry')
        instrument = create_nx_group(entry, 'instrument', 'NXinstrument')
        create_dataset(instrument, 'name', 'RECTANGLE', {'short_name': 'RECT'})
        sample = create_nx_group(entry, 'sample', 'NXsample')
        source = create_nx_group(instrument, 'source', 'NXsource')
        create_dataset(source, "depends_on", "/entry/instrument/source/transformations/beam_direction_offset")
        add_source_position(source)
        detector = add_detector(instrument, 'detector')
