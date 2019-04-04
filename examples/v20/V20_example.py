from collections import OrderedDict
from nexusutils.nexusbuilder import NexusBuilder
import h5py
import numpy as np
import nexusformat.nexus as nexus
from nexusjson.nexus_to_json import NexusToDictConverter, create_writer_commands, object_to_json_file
from datetime import datetime

airbus_choppers = [1, 2, 6, 7]
julich_choppers = [3, 4, 5, 8]


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


def __copy_existing_data(downscale_detecter=False):
    """
    Copy data from the existing NeXus file
    """
    raw_event_path = nx_entry_name + '/instrument/detector_1/raw_event_data/'
    builder.add_nx_group(builder.get_root()['instrument/detector_1'], 'raw_event_data', 'NXevent_data')
    builder.copy_items(OrderedDict(
        [('entry-01/Delayline_events/event_time_offset', raw_event_path + 'event_time_offset')
         ]))

    __copy_and_transform_dataset(builder.source_file, 'entry-01/Delayline_events/event_index',
                                 raw_event_path + 'event_index', dtype=np.uint64)

    def shift_time(timestamps):
        first_timestamp = 59120017391465
        new_start_time = 1543584772000000000
        return timestamps - first_timestamp + new_start_time

    event_time_zero_ds = __copy_and_transform_dataset(builder.source_file, 'entry-01/Delayline_events/event_time_zero',
                                                      raw_event_path + 'event_time_zero', shift_time)
    event_time_zero_ds.attrs.create('units', np.array('ns').astype('|S2'))
    event_time_zero_ds.attrs.create('offset', np.array('1970-01-01T00:00:00').astype('|S19'))

    def downscale_detector_resolution(ids):
        original_res = (2 ** 16) ** 2
        target_res = 150 ** 2
        scale_factor = target_res / original_res
        return (ids * scale_factor).astype(np.uint32)

    if downscale_detecter:
        __copy_and_transform_dataset(builder.source_file, 'entry-01/Delayline_events/event_id',
                                     raw_event_path + 'event_id', downscale_detector_resolution)
    else:
        __copy_and_transform_dataset(builder.source_file, 'entry-01/Delayline_events/event_id',
                                     raw_event_path + 'event_id')


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

    builder.add_dataset(chopper_group, 'speed', [0])
    builder.add_dataset(chopper_group, 'speed_setpoint', [0])
    builder.add_dataset(chopper_group, 'phase', [0])
    builder.add_dataset(chopper_group, 'phase_setpoint', [0])
    builder.add_dataset(chopper_group, 'factor', [0])

    tdc_log = builder.add_nx_group(chopper_group, 'top_dead_center', 'NXlog')


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
    builder.add_dataset(detector_group, 'detector_number', pixel_ids)

    # builder.add_shape(detector_group, 'detector_shape', vertices, faces, detector_faces.T)
    # Add detector position
    transforms = builder.add_nx_group(detector_group, 'transformations', 'NXtransformations')
    orientation = builder.add_transformation(transforms, 'rotation', [90.0], 'deg', [0.0, 1.0, 0.0], name='orientation',
                                             depends_on='.')
    z_offset = builder.add_transformation(transforms, 'translation', [0.049], 'm', [0.0, 0.0, -1.0],
                                          name='beam_direction_offset', depends_on=orientation.name)
    x_offset = builder.add_transformation(transforms, 'translation', [0.971], 'm', [1.0, 0.0, 0.0], name='location',
                                          depends_on=z_offset.name)
    builder.add_dataset(detector_group, 'depends_on', x_offset.name)

    # Placeholders for streamed data
    for channel_number in range(4):
        builder.add_nx_group(detector_group, f'waveforms_channel_{channel_number}', 'NXlog')
        builder.add_nx_group(detector_group, f'pulses_channel_{channel_number}', 'NXlog')

    for hv_power_supply_channel in range(4):
        builder.add_nx_group(detector_group, f'hv_supply_voltage_channel_{hv_power_supply_channel + 1}', 'NXlog')
        builder.add_nx_group(detector_group, f'hv_supply_current_channel_{hv_power_supply_channel + 1}', 'NXlog')
        builder.add_nx_group(detector_group, f'hv_supply_status_channel_{hv_power_supply_channel + 1}', 'NXlog')

    builder.add_nx_group(builder.get_root(), 'raw_event_data', 'NXevent_data')


def __add_monitors(builder):
    """
    Helium-3 monitor
    :param builder:
    :return:
    """
    distance_from_sample = -3.298
    monitor_group_1 = builder.add_nx_group(builder.get_root(), 'monitor_1', 'NXmonitor')
    builder.add_nx_group(monitor_group_1, 'events', 'NXevent_data')
    builder.add_nx_group(monitor_group_1, 'waveforms', 'NXlog')
    builder.add_dataset(monitor_group_1, 'detector_id', 90000)
    monitor_1_transforms = builder.add_nx_group(monitor_group_1, 'transformations', 'NXtransformations')
    monitor_1_z_offset = builder.add_transformation(monitor_1_transforms, 'translation', [distance_from_sample], 'm',
                                                    [0.0, 0.0, 1.0])
    builder.add_dataset(monitor_group_1, 'depends_on', monitor_1_z_offset.name)
    builder.add_dataset(monitor_group_1, 'name', 'Helium-3 monitor')


def __add_readout_system(builder):
    for readout_system_number in ('1', '2'):
        group_name = f'readout_system_{readout_system_number}'
        readout_group = builder.get_root().create_group(group_name)
        builder.add_nx_group(readout_group, 's_diff', 'NXlog')
        builder.add_nx_group(readout_group, 'n_diff', 'NXlog')
        builder.add_nx_group(readout_group, 'status', 'NXlog')


def __create_file_writer_command(filepath):
    streams = {}

    # DENEX detector
    detector_topic = 'denex_detector'
    __add_data_stream(streams, detector_topic, 'delay_line_detector',
                      '/entry/instrument/detector_1/raw_event_data', 'ev42')
    detector_debug_topic = 'denex_debug'
    for detector_channel in range(4):
        __add_data_stream(streams, detector_debug_topic, f'Denex_Adc0_Ch{detector_channel}',
                          f'/entry/instrument/detector_1/pulses_channel_{detector_channel}', 'ev42')
        __add_data_stream(streams, detector_debug_topic, f'Denex_Adc0_Ch{detector_channel}_waveform',
                          f'/entry/instrument/detector_1/waveforms_channel_{detector_channel}', 'senv')

    # Detector HV supply
    hv_supply_topic = 'V20_detectorPower'
    for hv_power_supply_channel in range(4):
        __add_data_stream(streams, hv_supply_topic, f'	HZB-V20:Det-PwrC-01:02:00{hv_power_supply_channel}:VMon',
                          f'/entry/instrument/detector_1/hv_supply_voltage_channel_{hv_power_supply_channel + 1}',
                          'f142', 'double')
        __add_data_stream(streams, hv_supply_topic, f'	HZB-V20:Det-PwrC-01:02:00{hv_power_supply_channel}:IMon',
                          f'/entry/instrument/detector_1/hv_supply_current_channel_{hv_power_supply_channel + 1}',
                          'f142', 'double')
        __add_data_stream(streams, hv_supply_topic, f'	HZB-V20:Det-PwrC-01:02:00{hv_power_supply_channel}:Pw',
                          f'/entry/instrument/detector_1/hv_supply_status_channel_{hv_power_supply_channel + 1}',
                          'f142', 'int32')

    # Monitors
    monitor_topic = 'monitor'
    __add_data_stream(streams, monitor_topic, 'Monitor_Adc0_Ch1',
                      '/entry/monitor_1/events', 'ev42')
    __add_data_stream(streams, monitor_topic, 'Monitor_Adc0_Ch1',
                      '/entry/monitor_1/waveforms', 'senv')

    # Choppers
    chopper_topic = 'V20_choppers'
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0401:TDC_array',
                      '/entry/instrument/chopper_1/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0402:TDC_array',
                      '/entry/instrument/chopper_2/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0101:TDC_array',
                      '/entry/instrument/chopper_3/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0102:TDC_array',
                      '/entry/instrument/chopper_4/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0301:TDC_array',
                      '/entry/instrument/chopper_5/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0501:TDC_array',
                      '/entry/instrument/chopper_6/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0502:TDC_array',
                      '/entry/instrument/chopper_7/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0302:TDC_array',
                      '/entry/instrument/chopper_8/top_dead_center', 'senv')
    __add_data_stream(streams, chopper_topic, 'HZB-V20:Chop-Drv-0101:Ref_Unix_asub.VALF',
                      '/entry/instrument/chopper_3/ntp_to_mrf_comparison', 'f142', 'int32')

    # Readout system timing status
    timing_status_topic = 'V20_timingStatus'
    for readout_system_number in ('1', '2'):
        group_name = f'readout_system_{readout_system_number}'
        __add_data_stream(streams, timing_status_topic, f'HZB-V20:TS-RO{readout_system_number}:TS-SDiff-RBV',
                          f'/entry/{group_name}/s_diff', 'f142', 'double')
        __add_data_stream(streams, timing_status_topic, f'HZB-V20:TS-RO{readout_system_number}:TS-NDiff-RBV',
                          f'/entry/{group_name}/n_diff', 'f142', 'double')
        __add_data_stream(streams, timing_status_topic, f'HZB-V20:TS-RO{readout_system_number}:STATUS2-RBV',
                          f'/entry/{group_name}/status', 'f142', 'int32')

    # Linear stages
    linear_motion_topic = 'V20_linearStages'
    for axis in ('1', '2'):
        group_name = f'linear_axis_{axis}'
        __add_data_stream(streams, linear_motion_topic, f'SES-PREMP:MC-MCU-01:m1{axis}.VAL',
                          f'/entry/{group_name}/target_value', 'f142', 'double')
        __add_data_stream(streams, linear_motion_topic, f'SES-PREMP:MC-MCU-01:m1{axis}.RBV',
                          f'/entry/{group_name}/value', 'f142', 'double')
        __add_data_stream(streams, linear_motion_topic, f'SES-PREMP:MC-MCU-01:m1{axis}.STAT',
                          f'/entry/{group_name}/status', 'f142', 'int32')

    # event_data_link = {'name': 'raw_event_data',
    #                   'target': '/entry/instrument/detector_1/raw_event_data'}
    # links = {'/entry/raw_event_data': event_data_link}
    links = {}

    converter = NexusToDictConverter()
    nexus_file = nexus.nxload(filepath)
    tree = converter.convert(nexus_file, streams, links)
    # The Kafka broker at V20 is v20-udder1, but due to the network setup at V20 we have to use the IP: 192.168.1.80
    # Use a timestamp in the output filename, but avoid characters "-" and ":"
    iso8601_str_seconds = datetime.now().isoformat().split('.')[0]
    timestamp = iso8601_str_seconds.replace(':', '_')
    timestamp = timestamp.replace('-', '_')
    write_command, stop_command = create_writer_commands(tree,
                                                         '/data/kafka-to-nexus/FILENAME',  # NICOS replaces FILENAME
                                                         broker='192.168.1.80:9092',
                                                         start_time='STARTTIME')  # NICOS replaces STARTTIME
    object_to_json_file(write_command, 'V20_file_write_start.json')
    object_to_json_file(stop_command, 'V20_file_write_stop.json')


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


if __name__ == '__main__':
    output_filename = 'V20_example_11.nxs'
    input_filename = 'adc_test8_half_cover_w_waveforms.nxs'  # None
    nx_entry_name = 'entry'
    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, input_nexus_filename=input_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        instrument_group = builder.add_instrument('V20', 'instrument')
        builder.add_user('Many people', 'ESS, HZB, STFC', number=1)
        __add_detector(builder)
        __add_choppers(builder)
        __add_monitors(builder)
        sample_group = builder.add_sample()
        builder.add_dataset(sample_group, 'description',
                            'hBN target with 1.0 mm diameter hole')
        #builder.add_dataset(sample_group, 'description',
        #                    'hBN target with 0.2 mm diameter hole')

        # Add a source at the position of the first chopper
        builder.add_source('V20_14hz_chopper_source', 'source', [0.0, 0.0, -50.598+21.7])

        # Add start_time dataset (required by Mantid)
        iso8601_str_seconds = datetime.now().isoformat().split('.')[0]
        builder.add_dataset(builder.get_root(), 'start_time', iso8601_str_seconds)

        # Copy event data into detector
        __copy_existing_data()

        # TODO Add guides, shutters, any other known components
        #   Add more details on the sample

        # Notes on geometry:

        # Geometry is altered slightly from reality such that analysis does not require handling the curved guides
        # and can treat the neutron paths as straight lines between source and sample, and sample and detector.

        # Since we will use timestamps from the first (furthest from detector) chopper as the pulse timestamps,
        # the "source" is placed at the position of the first chopper

        # kafkacat -b 192.168.1.80 -t V20_writerCommand -X message.max.bytes=20000000 V20_file_write_stop.json -P

    __create_file_writer_command(output_filename)
