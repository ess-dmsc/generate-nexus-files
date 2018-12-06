from collections import OrderedDict
from nexusutils.nexusbuilder import NexusBuilder
import h5py
import numpy as np
import nexusformat.nexus as nexus
from nexusjson.nexus_to_json import NexusToDictConverter, create_writer_commands, object_to_json_file


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


def __copy_existing_data():
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

    __copy_and_transform_dataset(builder.source_file, 'entry-01/Delayline_events/event_id',
                                 raw_event_path + 'event_id', downscale_detector_resolution)


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


def __add_chopper(builder, name, speed=None):
    chopper_group = builder.add_nx_group(instrument_group, name, 'NXdisk_chopper')
    if speed is not None:
        builder.add_dataset(chopper_group, 'rotation_speed', speed, {'units': 'Hz'})
    unix_log = builder.add_nx_group(chopper_group, 'top_dead_centre_unix', 'NXlog')
    return unix_log


def __add_choppers(builder):
    unix_log = __add_chopper(builder, 'chopper_1', 14.0)
    __add_chopper(builder, 'chopper_2')
    __add_chopper(builder, 'chopper_3')
    __add_chopper(builder, 'chopper_4')
    __add_chopper(builder, 'chopper_5')
    __add_chopper(builder, 'chopper_6')
    __add_chopper(builder, 'chopper_7')
    __add_chopper(builder, 'chopper_8')

    def shift_time(timestamps):
        first_timestamp = 1542008231816585559
        new_start_time = 1543584772000000000
        return timestamps - first_timestamp + new_start_time

    with h5py.File('chopper_tdc_file.hdf', 'r') as chopper_file:
        __copy_and_transform_dataset(chopper_file, 'entry-01/ca_unix_double/time', unix_log.name + '/time', shift_time)
        builder._NexusBuilder__copy_dataset(chopper_file['entry-01/ca_unix_double/value'], unix_log.name + '/value')
        unix_log['time'].attrs.create('units', 'ns', dtype='|S2')


def __add_detector(builder):
    # Add description of V20's DENEX (delay line) detector

    pixels_per_axis = 150  # 65535 (requires int64)
    pixel_size = 0.002
    half_detector_width = 0.15
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

    pixel_ids = np.arange(0, 22500, 1, dtype=int)
    pixel_ids = np.reshape(pixel_ids, (150, 150))
    builder.add_dataset(detector_group, 'detector_number', pixel_ids)

    # builder.add_shape(detector_group, 'detector_shape', vertices, faces, detector_faces.T)
    # Add detector position
    # detector_transformations = builder.add_nx_group(detector_group, 'transformations', 'NXtransformations')
    location_dataset = builder.add_transformation(detector_group,
                                                  'translation', [5.0], 'm', [0.0, 0.0, 1.0], name='location')
    builder.add_dataset(detector_group, 'depends_on', location_dataset.name)

    builder.add_nx_group(detector_group, 'waveform_data_1', 'NXlog')
    builder.add_nx_group(detector_group, 'waveform_data_2', 'NXlog')

    #builder.add_nx_group(builder.get_root(), 'raw_event_data', 'NXevent_data')


def __add_users(builder):
    user_names = ['Tobias Richter', 'Jonas Nilsson', 'Nicklas Holmberg', 'Irina Stefanescu', 'Gregor Nowak',
                  'Neil Vaytet', 'Torben Nielsen', 'Andrew Jackson', 'Vendula Maulerova']
    roles = ['Project Owner', 'Detector and Monitor DAQ', 'Timing system', 'DG contact', 'BEER detector team',
             'Mantid Reduction, WFM treatment', 'Mantid/McStas', 'Observer', 'Monitor tests']
    __add_user_group(builder, user_names, roles, 'ESS')

    user_names = ['Peter Kadletz', 'Robin Woracek']
    roles = ['Beamline Responsible', 'Beamline Responsible']
    __add_user_group(builder, user_names, roles, 'HZB')

    user_names = ['Michael Hart', 'Matthew Jones', 'Owen Arnold', 'Will Smith']
    roles = ['V20 NICOS', 'Streaming', 'Mantid', 'Set-up of timing system']
    __add_user_group(builder, user_names, roles, 'STFC')


def __add_user_group(builder, user_names, roles, institution):
    users = builder.add_nx_group(builder.get_root(), institution + '_users', 'NXuser')
    user_names_ascii = [n.encode('ascii', 'ignore') for n in user_names]
    roles_ascii = [n.encode('ascii', 'ignore') for n in roles]
    users.create_dataset('name', (len(user_names_ascii),), '|S' + str(len(max(user_names_ascii, key=len))),
                         user_names_ascii)
    users.create_dataset('role', (len(roles_ascii),), '|S' + str(len(max(roles_ascii, key=len))),
                         roles_ascii)
    builder.add_dataset(users, 'affiliation', institution)


def __add_monitors(builder):
    monitor_group_1 = builder.add_nx_group(instrument_group, 'monitor_1', 'NXmonitor')
    builder.add_nx_group(monitor_group_1, 'raw_event_data', 'NXevent_data')
    builder.add_nx_group(monitor_group_1, 'waveform_data', 'NXlog')
    builder.add_dataset(monitor_group_1, 'detector_id', 22500)
    monitor_group_2 = builder.add_nx_group(instrument_group, 'monitor_2', 'NXmonitor')
    builder.add_nx_group(monitor_group_2, 'raw_event_data', 'NXevent_data')
    builder.add_nx_group(monitor_group_2, 'waveform_data', 'NXlog')
    builder.add_dataset(monitor_group_2, 'detector_id', 22501)


def __create_file_writer_command(filepath):
    streams = {}
    __add_data_stream(streams, 'V20_rawEvents', 'delay_line_detector',
                      '/entry/instrument/detector_1/raw_event_data', 'ev42')
    __add_data_stream(streams, 'V20_waveforms', 'delay_line_detector_wf',  # different source name due to DM-1129 (JIRA)
                      '/entry/instrument/detector_1/waveform_data_1', 'senv')
    __add_data_stream(streams, 'V20_waveforms', 'delay_line_detector_wf',
                      '/entry/instrument/detector_1/waveform_data_2', 'senv')

    __add_data_stream(streams, 'V20_rawEvents', 'monitor_1',
                      '/entry/instrument/monitor_1/raw_event_data', 'ev42')
    __add_data_stream(streams, 'V20_waveforms', 'monitor_1_wf',
                      '/entry/instrument/monitor_1/waveform_data', 'senv')
    __add_data_stream(streams, 'V20_rawEvents', 'monitor_2',
                      '/entry/instrument/monitor_2/raw_event_data', 'ev42')
    __add_data_stream(streams, 'V20_waveforms', 'monitor_2_wf',
                      '/entry/instrument/monitor_2/waveform_data', 'senv')
    for chopper_number in range(1, 9):
        suffix = '_A' if chopper_number in [1, 2, 6, 7] else '_J'  # labels if Airbus or Julich chopper
        __add_data_stream(streams, 'V20_choppers', 'chopper_' + str(chopper_number) + suffix,
                          '/entry/instrument/chopper_' + str(chopper_number) + '/top_dead_centre_unix', 'f142')

    for pv in lakeshore_pvs:
        log_name = pv.split(':')[-1]
        __add_data_stream(streams, 'V20_logs', pv,
                          '/entry/instrument/temperature_controller/' + log_name, 'f142')

    #event_data_link = {'name': 'raw_event_data',
    #                   'target': '/entry/instrument/detector_1/raw_event_data'}
    #links = {'/entry/raw_event_data': event_data_link}

    converter = NexusToDictConverter()
    nexus_file = nexus.nxload(filepath)
    tree = converter.convert(nexus_file, streams, links)
    # The Kafka broker at V20 is v20-udder1, but probably need to use the IP: 192.168.1.80
    write_command, stop_command = create_writer_commands(tree, 'V20_example_output.nxs', broker='192.168.1.80:9092')
    object_to_json_file(write_command, 'V20_file_write_start.json')
    object_to_json_file(stop_command, 'V20_file_write_stop.json')


def __add_data_stream(streams, topic, source, path, module):
    options = {
        'topic': topic,
        'source': source,
        'writer_module': module
    }
    streams[path] = options


def __add_sample_env_device(group_name, name, description=None):
    env_group = builder.add_nx_group(builder.get_root()['instrument'], group_name, 'NXenvironment')
    builder.add_dataset(env_group, 'name', name)
    if description is not None:
        builder.add_dataset(env_group, 'description', description)
    return env_group


if __name__ == '__main__':
    output_filename = 'V20_example_10.nxs'
    input_filename = 'adc_test8_half_cover_w_waveforms.nxs'  # None
    nx_entry_name = 'entry'
    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, input_nexus_filename=input_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        instrument_group = builder.add_instrument('V20', 'instrument')
        __add_users(builder)
        __add_detector(builder)
        __add_choppers(builder)
        __add_monitors(builder)
        sample_group = builder.add_sample()
        builder.add_dataset(sample_group, 'description',
                            'We\'re not sure what it is, but it glows with a mysterious green light...')

        # Add a source at the position of the first chopper
        builder.add_source('V20_14hz_chopper_source', 'source', [0.0, 0.0, -24.5])

        # Copy event data into detector
        __copy_existing_data()

        lakeshore_group = __add_sample_env_device('temperature_controller', 'Lakeshore Huginn',
                                                  'Lakeshore Huginn temperature controller')
        lakeshore_pvs = [
            'Huginn:SUBCRYO_READYTOMEAS',
            'Huginn:SUBCRYO_SETP',
            'Huginn:SUBCRYO_SETP_S',
            'Huginn_LS:INNAME0',
            'Huginn_LS:INNAME1',
            'Huginn_LS:INNAME2',
            'Huginn_LS:INNAME3',
            'Huginn_LS:KRDG0',
            'Huginn_LS:KRDG1',
            'Huginn_LS:KRDG2',
            'Huginn_LS:KRDG3',
            'Huginn:MAINCRYO_SETP',
            'Huginn:SUBCRYO_ONOFF',
            'Huginn:SUBCRYO_SETP_MAX'
        ]
        for pv in lakeshore_pvs:
            log_name = pv.split(':')[-1]
            builder.add_nx_group('/entry/instrument/temperature_controller/', log_name, 'NXlog')

        # TODO Add guides, shutters, any other known components
        #   Add more details on the sample

        # Notes on geometry:

        # Geometry is altered slightly from reality such that analysis does not require handling the curved guides
        # and can treat the neutron paths as straight lines between source and sample, and sample and detector.

        # Since we will use timestamps from the first (furthest from detector) chopper as the pulse timestamps,
        # the "source" is placed at the position of the first chopper

    __create_file_writer_command(output_filename)
