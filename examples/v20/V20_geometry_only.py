from nexusutils.nexusbuilder import NexusBuilder
import numpy as np

airbus_choppers = [1, 2, 6, 7]
julich_choppers = [3, 4, 5, 8]


def __add_chopper(builder, number):
    chopper_group = builder.add_nx_group(instrument_group, 'chopper_' + str(number), 'NXdisk_chopper')
    if number in julich_choppers:
        builder.add_dataset(chopper_group, 'speed', [0])
        builder.add_dataset(chopper_group, 'speed_setpoint', [0])
        builder.add_dataset(chopper_group, 'phase', [0])
        builder.add_dataset(chopper_group, 'phase_setpoint', [0])
        builder.add_dataset(chopper_group, 'factor', [0])


def __add_choppers(builder):
    __add_chopper(builder, 1)
    for chopper_number in range(2, 9):
        __add_chopper(builder, chopper_number)


def __add_detector(builder):
    # Add description of V20's DENEX (delay line) detector

    pixels_per_axis = 300  # 65535 (requires int64)
    pixel_size = 0.002
    half_detector_width = 0.3
    half_pixel_width = 0.002 / 2.0
    single_axis_offsets = (pixel_size * np.arange(0, pixels_per_axis, 1,
                                                  dtype=np.float)) - half_detector_width + half_pixel_width
    detector_group = builder.add_nx_group(builder.get_root()['instrument'], 'detector_1', 'NXdetector')
    x_offsets, y_offsets = np.meshgrid(single_axis_offsets,
                                       single_axis_offsets)
    builder.add_dataset(detector_group, 'x_pixel_offset', x_offsets, {'units': 'm'})
    builder.add_dataset(detector_group, 'y_pixel_offset', y_offsets, {'units': 'm'})

    builder.add_dataset(detector_group, 'local_name', 'DENEX delay line detector')

    pixel_shape = builder.add_nx_group(detector_group, 'pixel_shape', 'NXoff_geometry')
    pixel_verts = np.array([[-0.001, -0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0], [-0.001, 0.001, 0.0]],
                           dtype=np.float32)
    pixel_winding_order = np.array([0, 1, 2, 3], dtype=np.int32)
    pixel_faces = np.array([0], dtype=np.int32)
    builder.add_dataset(pixel_shape, 'faces', pixel_faces)
    builder.add_dataset(pixel_shape, 'vertices', pixel_verts, {'units': 'm'})
    builder.add_dataset(pixel_shape, 'winding_order', pixel_winding_order)

    pixel_ids = np.arange(0, pixels_per_axis ** 2, 1, dtype=int)
    pixel_ids = np.reshape(pixel_ids, (pixels_per_axis, pixels_per_axis))
    # transpose pixel_ids to get the correct physical orientation of the detector
    # this was discovered by looking at measured data
    builder.add_dataset(detector_group, 'detector_number', pixel_ids.transpose())

    # Add detector position
    transforms = builder.add_nx_group(detector_group, 'transformations', 'NXtransformations')
    z_offset = builder.add_transformation(transforms, 'translation', [0.049], 'm', [0.0, 0.0, -1.0],
                                          name='beam_direction_offset', depends_on='.')
    x_offset = builder.add_transformation(transforms, 'translation', [0.971], 'm', [1.0, 0.0, 0.0], name='location',
                                          depends_on=z_offset.name)
    orientation = builder.add_transformation(transforms, 'rotation', [np.pi/2.0], 'rad', [0.0, 1.0, 0.0], name='orientation',
                                             depends_on=x_offset.name)
    builder.add_dataset(detector_group, 'depends_on', orientation.name)

    for channel_number in range(4):
        builder.add_nx_group(detector_group, 'waveform_data_' + str(channel_number), 'NXlog')
        builder.add_nx_group(detector_group, 'pulse_events_' + str(channel_number), 'NXlog')
    builder.add_nx_group(builder.get_root(), 'timing_system_waveform_data', 'NXlog')


def __add_monitors(builder):
    monitor_group_1 = builder.add_nx_group(builder.get_root(), 'monitor_1', 'NXmonitor')
    builder.add_dataset(monitor_group_1, 'detector_id', 90000)
    monitor_1_transforms = builder.add_nx_group(monitor_group_1, 'transformations', 'NXtransformations')
    monitor_1_z_offset = builder.add_transformation(monitor_1_transforms, 'translation', [-3.298], 'm', [0.0, 0.0, 1.0])
    builder.add_dataset(monitor_group_1, 'depends_on', monitor_1_z_offset.name)
    monitor_group_2 = builder.add_nx_group(builder.get_root(), 'monitor_2', 'NXmonitor')
    builder.add_dataset(monitor_group_2, 'detector_id', 90001)


def __add_data_stream(streams, topic, source, path, module, type=None):
    options = {
        'topic': topic,
        'source': source,
        'writer_module': module
    }
    if type is not None:
        options['type'] = type
    streams[path] = options


if __name__ == '__main__':
    output_filename = 'V20_geometry.nxs'
    nx_entry_name = 'entry'
    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        instrument_group = builder.add_instrument('V20', 'instrument')
        __add_detector(builder)
        __add_choppers(builder)
        __add_monitors(builder)
        sample_group = builder.add_sample()
        builder.add_dataset(sample_group, 'description',
                            'The sample')

        # Add a source at the position of the first chopper
        builder.add_source('V20_14hz_chopper_source', 'source', [0.0, 0.0, -50.598 + 21.7])

        # TODO Add shutters?

        # Notes on geometry:

        # Geometry is altered slightly from reality such that analysis does not require handling the curved guides
        # and can treat the neutron paths as straight lines between source and sample, and sample and detector.

        # Since we will use timestamps from the first (furthest from detector) chopper as the pulse timestamps,
        # the "source" is placed at the position of the first chopper
