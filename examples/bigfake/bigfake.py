from collections import OrderedDict
from nexusutils.nexusbuilder import NexusBuilder
from nexusutils.detectorplotter import DetectorPlotter

import numpy as np
import nexusformat.nexus as nexus
from nexusjson.nexus_to_json import NexusToDictConverter, create_writer_commands, object_to_json_file
from datetime import datetime
from typing import List


def __copy_and_transform_dataset(source_file, source_path, target_path, transformation=None, dtype=None):
    source_data = source_file[source_path][...]
    if transformation is not None:
        transformed_data = transformation(source_data)
    else:
        transformed_data = source_data
    if dtype is None:
        dtype = transformed_data.dtype
    target_dataset = builder.target_file.create_dataset(target_path, transformed_data.shape,
                                                        dtype=dtype,
                                                        compression=builder.compress_type,
                                                        compression_opts=builder.compress_opts)
    target_dataset[...] = transformed_data
    return target_dataset

def __copy_log(builder, source_group, destination_group, nx_component_class=None):
    split_destination = destination_group.split('/')
    log_name = split_destination[-1]
    if nx_component_class is not None:
        component_name = split_destination[-2]
        parent_path_from_entry = '/'.join(split_destination[1:-2])
        component_group = builder.add_nx_group(parent_path_from_entry, component_name, nx_component_class)
        builder.add_nx_group(component_group, log_name, 'NXlog')
    else:
        builder.add_nx_group('/'.join(split_destination[1:-1]), log_name, 'NXlog')
    builder.copy_items(OrderedDict(
        [(source_group + '/time', destination_group + '/time'),
         (source_group + '/value', destination_group + '/value')]))


def __add_chopper(builder, number):
    chopper_group = builder.add_nx_group(instrument_group, 'chopper_' + str(number), 'NXdisk_chopper')

    # TDC position is unknown for the Airbus choppers, due to them being refurbished two times
    if number is 1:
        builder.add_dataset(chopper_group, 'name', 'Airbus, Source Chopper, ESS Pulse, Disc 1')
        builder.add_dataset(chopper_group, 'slit_edges', [0., 23.], attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 1)
        builder.add_dataset(chopper_group, 'slit_height', 150., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 350., attributes={'units': 'mm'})
        distance_from_sample = -27.4
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 2:
        builder.add_dataset(chopper_group, 'name', 'Airbus, Source Chopper, ESS Pulse, Disc 2')
        builder.add_dataset(chopper_group, 'slit_edges', [0., 50.], attributes={'units': 'deg'})
        # Actually has 2 slits, but only one is used and I don't have dimensions for the second slit
        builder.add_dataset(chopper_group, 'slits', 1)
        builder.add_dataset(chopper_group, 'slit_height', 150., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 350., attributes={'units': 'mm'})
        distance_from_sample = -27.4
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 3:
        builder.add_dataset(chopper_group, 'name', 'Juelich, WFM Chopper, Disc 1')
        builder.add_dataset(chopper_group, 'slit_edges', np.array(
            [0., 83.71, 94.7, 140.49, 155.79, 193.26, 212.56, 242.32, 265.33, 287.91, 314.37, 330.3]) + 15.0,
                            attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 6)
        builder.add_dataset(chopper_group, 'slit_height', 130., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 300., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'ntp_to_mrf_comparison', 0)
        distance_from_sample = -20.55
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 4:
        builder.add_dataset(chopper_group, 'name', 'Juelich, WFM Chopper, Disc 2')
        builder.add_dataset(chopper_group, 'slit_edges', np.array(
            [0., 65.04, 76.03, 126.1, 141.4, 182.88, 202.18, 235.67, 254.97, 284.73, 307.74, 330.0]) + 15.0,
                            attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 6)
        builder.add_dataset(chopper_group, 'slit_height', 130., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 300., attributes={'units': 'mm'})
        distance_from_sample = -20.55
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 5:
        builder.add_dataset(chopper_group, 'name', 'Juelich, Frame Overlap Chopper, Disc 1')
        builder.add_dataset(chopper_group, 'slit_height', 130., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 300., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'slit_edges', np.array(
            [0., 64.35, 84.99, 125.05, 148.29, 183.41, 205.22, 236.4, 254.27, 287.04, 302.8, 335.53]) + 15.0,
                            attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 6)
        distance_from_sample = -18.6
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 6:
        builder.add_dataset(chopper_group, 'name', 'Airbus, Wavelength-Band Chopper, Disc 1')
        builder.add_dataset(chopper_group, 'pair_separation', 24.2, attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'slit_edges', [0., 140.], attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 1)
        builder.add_dataset(chopper_group, 'slit_height', 150., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 350., attributes={'units': 'mm'})
        distance_from_sample = -17.4
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 7:
        builder.add_dataset(chopper_group, 'name', 'Airbus, Wavelength-Band Chopper, Disc 2')
        builder.add_dataset(chopper_group, 'pair_separation', 24.2, attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'slit_edges', [0., 202.], attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 1)
        builder.add_dataset(chopper_group, 'slit_height', 150., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 350., attributes={'units': 'mm'})
        distance_from_sample = -11.5
        record_chopper_position(builder, chopper_group, distance_from_sample)
    elif number is 8:
        builder.add_dataset(chopper_group, 'name', 'Juelich, Frame Overlap Chopper, Disc 2')
        builder.add_dataset(chopper_group, 'slit_height', 130., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'radius', 300., attributes={'units': 'mm'})
        builder.add_dataset(chopper_group, 'slit_edges', np.array(
            [0., 79.78, 116.38, 136.41, 172.47, 191.73, 221.94, 240.81, 267.69, 287.13, 311.69, 330.89]) + 15.0,
                            attributes={'units': 'deg'})
        builder.add_dataset(chopper_group, 'slits', 6)
        distance_from_sample = -11.5
        record_chopper_position(builder, chopper_group, distance_from_sample)

    builder.add_feature("B89B086951FEFDDF")
    add_nxlog(builder, 'top_dead_center', parent_path=chopper_group.name, number_of_cues=100, units="ns")
    add_nxlog(builder, 'speed', parent_path=chopper_group.name, number_of_cues=10, units="Hz")
    add_nxlog(builder, 'delay', parent_path=chopper_group.name, number_of_cues=10, units="ms")


def record_chopper_position(builder, chopper_group, distance_from_sample):
    transforms = builder.add_nx_group(chopper_group, 'transformations', 'NXtransformations')
    position = builder.add_transformation(transforms, 'translation', distance_from_sample, 'm', [0., 0., 1.],
                                          name='position')
    builder.add_dataset(chopper_group, 'depends_on', position.name)


def __add_choppers(builder):
    for chopper_number in range(1, 9):
        __add_chopper(builder, chopper_number)


def __add_detector(builder):
    """
    Description of V20's DENEX (delay line) detector
    :param builder:
    :return:
    """

    pixels_per_axis = 300  # 65535 (requires int64)
    pixel_size = 0.002
    half_detector_width = 0.3
    half_pixel_width = pixel_size / 2.0
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
    builder.add_dataset(detector_group, 'detector_number', pixel_ids)

    # builder.add_shape(detector_group, 'detector_shape', vertices, faces, detector_faces.T)
    # Add detector position
    transforms = builder.add_nx_group(detector_group, 'transformations', 'NXtransformations')
    orientation = builder.add_transformation(transforms, 'rotation', [30.0], 'deg', [0.0, 1.0, 0.0], name='orientation',
                                             depends_on='.')
    z_offset = builder.add_transformation(transforms, 'translation', [0.049], 'm', [0.0, 0.0, -1.0],
                                          name='beam_direction_offset', depends_on=orientation.name)
    x_offset = builder.add_transformation(transforms, 'translation', [0.971], 'm', [1.0, 0.0, 0.0], name='location',
                                          depends_on=z_offset.name)
    builder.add_dataset(detector_group, 'depends_on', x_offset.name)

    __add_readout_system(builder, detector_group)

    builder.add_fake_event_data(1000, 100)


def __add_monitors(builder):
    """
    Helium-3 monitor
    :param builder:
    :return:
    """
    distance_from_sample = -3.298
    monitor_group_1 = builder.add_nx_group(builder.get_root(), 'monitor_1', 'NXmonitor')
    monitor_event_group = monitor_group_1.create_group('events')
    builder.add_dataset(monitor_event_group, 'time', np.random.rand(1000).astype('float32'), {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(monitor_event_group, 'cue_timestamp_zero', np.random.rand(50).astype('float32'),
                            {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(monitor_event_group, 'cue_index', np.random.rand(50).astype('int32'))

    monitor_1_transforms = builder.add_nx_group(monitor_group_1, 'transformations', 'NXtransformations')
    monitor_1_z_offset = builder.add_transformation(monitor_1_transforms, 'translation', [distance_from_sample], 'm',
                                                    [0.0, 0.0, 1.0])
    builder.add_dataset(monitor_group_1, 'depends_on', monitor_1_z_offset.name)
    builder.add_dataset(monitor_group_1, 'name', 'Helium-3 monitor')


def __add_readout_system(builder, parent_group):
    for readout_system_number in ('1', '2'):
        group_name = f'readout_system_{readout_system_number}'
        readout_group = parent_group.create_group(group_name)
        readout_group.create_group('s_diff')
        readout_group.create_group('n_diff')
        readout_group.create_group('status')


def __add_motion_devices(builder):
    def _add_motion(builder, group_names: List[str], start_number: int = 0, nx_class: str = 'NXpositioner',
                    pv_root: str = None):
        for group_number, group_name in enumerate(group_names):
            group = builder.add_nx_group(builder.get_root()['sample/transformations'], group_name, nx_class)
            group.create_group('target_value')
            group.create_group('value')
            group.create_group('status')
            group.create_group('velocity')
            if pv_root is not None:
                builder.add_dataset(group, 'controller_record', pv_root.format(group_number + start_number))

    _add_motion(builder, ['linear_stage', 'tilting_angle_1', 'tilting_angle_2'], 1, pv_root='TUD-SMI:MC-MCU-01:m{}.VAL')
    _add_motion(builder, ['omega1', 'omega_2', 'phi'], 10, pv_root='HZB-V20:MC-MCU-01:m{}.VAL')
    #_add_motion(builder, ['slit'], nx_class='NXslit')


def __create_file_writer_command(filepath):
    pass

def __add_data_stream(streams, topic, source, path, module, type=None):
    options = {
        'topic': topic,
        'source': source,
        'writer_module': module
    }
    if type is not None:
        options['type'] = type
    streams[path] = options


def __add_sample_env_device(group_name, name, description=None):
    env_group = builder.add_nx_group(builder.get_root()['instrument'], group_name, 'NXenvironment')
    builder.add_dataset(env_group, 'name', name)
    if description is not None:
        builder.add_dataset(env_group, 'description', description)
    return env_group


def __add_attributes(node, attributes):
    for key in attributes:
        if isinstance(attributes[key], str):
            # Since python 3 we have to treat strings like this
            node.attrs.create(key, np.array(attributes[key]).astype('|S' + str(len(attributes[key]))))
        else:
            node.attrs.create(key, np.array(attributes[key]))

def add_nxlog(builder, nxlogname, parent_path='/', number_of_cues=1000, units="m", factor=1, attributes={}):
    """
    Adds example NXlog class to the file
    """
    time = 0.0
    index = 0
    cue_timestamps = []
    cue_indices = []
    times = np.array([])
    values = np.array([])
    for cue_number in range(number_of_cues):
        number_of_samples = np.random.randint(number_of_cues * 10, number_of_cues * 20)
        cue_timestamps.append(time)
        cue_indices.append(index)
        time += 0.2 * number_of_cues + (np.random.rand() * 20)
        if cue_number > 0:
            values = np.hstack([values, np.sort(np.random.rand(number_of_samples) * (1/number_of_cues)) + values[-1]])
            times = np.hstack(
                (
                    times,
                    cue_timestamps[-1] + (np.sort(np.random.rand(number_of_samples)) * (time - cue_timestamps[-1]))))
        else:
            values = np.sort(np.random.rand(number_of_samples) * (1/number_of_cues) * factor) + 0.21
            times = np.sort(np.random.rand(number_of_samples)) * time
        index += number_of_samples

    # Create an NXlog group in the sample group
    data_group = builder.add_nx_group(parent_path, nxlogname, 'NXlog')
    builder.add_dataset(data_group, 'time', times.astype('float32'), {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(data_group, 'value', values.astype('float32'), {'units': units})
    builder.add_dataset(data_group, 'cue_timestamp_zero', np.array(cue_timestamps).astype('float32'),
                        {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(data_group, 'cue_index', np.array(cue_indices).astype('int32'))
    return data_group


if __name__ == '__main__':
    output_filename = 'bigfake.nxs'
    input_filename = None
    nx_entry_name = 'entry'
    iso_timestamp = datetime.now().isoformat()

    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, input_nexus_filename=input_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        builder.add_dataset(builder.root, 'title', 'example to demonstrate a prospective ESS instrument NeXus file')
        instrument_group = builder.add_instrument('bigfake', 'instrument')
        slit_group = builder.add_nx_group(instrument_group, 'slit1', 'NXslit')
        builder.add_dataset(slit_group, 'depends_on', 'transformations/x_offset')
        slit_transforms = builder.add_nx_group(slit_group, 'transformations', 'NXtransformations')
        xo = add_nxlog(builder, 'x_offset', parent_path=slit_transforms.name, number_of_cues=1, units="mm")
        attributes = {'vector': [1.0, 0.0, 0.0], 'transformation_type': np.string_('translation'),
                      'depends_on': np.string_("y_offset")}
        for name in attributes.keys():
            xo.attrs.create(name, attributes[name])
        yo = add_nxlog(builder, 'y_offset', parent_path=slit_transforms.name, number_of_cues=2, units="mm")
        attributes = {'vector': [0.0, 1.0, 0.0], 'transformation_type': np.string_('translation'),
                    'depends_on': np.string_("distance")}
        for name in attributes.keys():
            yo.attrs.create(name, attributes[name])
        builder.add_transformation(slit_transforms, 'translation', [-1810], 'mm', [0.0, 0.0, 1.0], name='distance',
                                             depends_on='.')
        add_nxlog(builder, 'x_gap', parent_path=slit_group.name, number_of_cues=1, units="mm")
        add_nxlog(builder, 'y_gap', parent_path=slit_group.name, number_of_cues=1, units="mm")



        # Sample
        sample_group = builder.add_sample()
        transforms = builder.add_nx_group(sample_group, 'transformations', 'NXtransformations')
        builder.add_dataset(sample_group, 'name', 'white powder')
        builder.add_dataset(sample_group, 'chemical_formula', 'C17H21NO4')
        builder.add_dataset(sample_group, 'mass', 30, {'units':'g'})
        add_nxlog(builder, 'temperature', parent_path=sample_group.name, number_of_cues=7, units="K")
        add_nxlog(builder, 'pressure', parent_path=sample_group.name, number_of_cues=2, units="MPa")

        # Add a source at the position of the first chopper
        builder.add_source('BER', 'source', [0.0, 0.0, -50.598 + 21.7])

        builder.add_user('Gareth Murphy', 'ESS', number=1)
        __add_detector(builder)
        __add_choppers(builder)
        __add_monitors(builder)
        __add_motion_devices(builder)

        # Add start_time dataset (required by Mantid)
        iso8601_str_seconds = datetime.now().isoformat().split('.')[0]
        builder.add_dataset(builder.get_root(), 'start_time', iso_timestamp)  # NICOS will replace 8601TIME

        # Notes on geometry:

        # Geometry is altered slightly from reality such that analysis does not require handling the curved guides
        # and can treat the neutron paths as straight lines between source and sample, and sample and detector.

        # Since we will use timestamps from the first (furthest from detector) chopper as the pulse timestamps,
        # the "source" is placed at the position of the first chopper

        # kafkacat -b 192.168.1.80 -t V20_writerCommand -X message.max.bytes=20000000 V20_file_write_stop.json -P

    with DetectorPlotter(output_filename, nx_entry_name) as plotter:
        #plotter.plot_pixel_positions()
        pass
