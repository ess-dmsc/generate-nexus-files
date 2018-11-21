from collections import OrderedDict
from nexusutils.nexusbuilder import NexusBuilder
from nexusutils.detectorplotter import DetectorPlotter
import h5py
import numpy as np


def __copy_existing_data():
    """
    Copy data from the existing NeXus file
    """
    raw_event_path = nx_entry_name + '/instrument/detector_1/raw_event_data/'
    builder.copy_items(OrderedDict(
        [('entry-01/Delayline_events', nx_entry_name + '/instrument/detector_1/raw_event_data'),
         ('entry-01/Delayline_events/event_id', raw_event_path + 'event_id'),
         ('entry-01/Delayline_events/event_index', raw_event_path + 'event_index'),
         ('entry-01/Delayline_events/event_time_offset', raw_event_path + 'event_time_offset'),
         ('entry-01/Delayline_events/event_time_zero', raw_event_path + 'event_time_zero')
         ]))


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


def __add_choppers(builder):
    # TODO Add choppers - use diagram for positions, numbered from source end of beamline
    chopper_group_1 = builder.add_nx_group(instrument_group, 'chopper_1', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_2', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_3', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_4', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_5', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_6', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_7', 'NXdisk_chopper')
    builder.add_nx_group(instrument_group, 'chopper_8', 'NXdisk_chopper')
    builder.add_dataset(chopper_group_1, 'rotation_speed', 14.0, {'units': 'Hz'})
    tdc_log = builder.add_nx_group(chopper_group_1, 'top_dead_centre', 'NXlog')
    with h5py.File('chopper_tdc_file.hdf', 'r') as chopper_file:
        builder._NexusBuilder__copy_dataset(chopper_file['entry-01/ca_epics_double/time'], tdc_log.name + '/time')
        builder._NexusBuilder__copy_dataset(chopper_file['entry-01/ca_epics_double/value'], tdc_log.name + '/value')
        tdc_log['time'].attrs.create('units', 'ns', dtype='|S2')


def __add_detector(builder):
    # Add description of V20's DENEX (delay line) detector

    pixels_per_axis = 150  # 65535 (requires int64)
    detector_ids = np.reshape(np.arange(0, pixels_per_axis ** 2, 1, np.int32), (pixels_per_axis, pixels_per_axis))
    single_axis_offsets = (0.002 * np.arange(0, pixels_per_axis, 1, dtype=np.float)) - 0.15
    x_pixel_offsets, y_pixel_offsets = np.meshgrid(single_axis_offsets, single_axis_offsets)
    offsets = np.reshape(np.arange(0, pixels_per_axis ** 2, 1, np.int64), (pixels_per_axis, pixels_per_axis))
    builder.add_detector('DENEX delay line detector', 1, detector_ids,
                         {'x_pixel_offset': x_pixel_offsets, 'y_pixel_offset': y_pixel_offsets},
                         x_pixel_size=0.002, y_pixel_size=0.002)


def __add_users(builder):
    user_group = builder.add_user('Tobias Richter', 'ESS', 1)
    builder.add_dataset(user_group, 'role', 'Project Owner')
    user_group = builder.add_user('Jonas Nilsson', 'ESS', 2)
    builder.add_dataset(user_group, 'role', 'Detector and Monitor DAQ')
    user_group = builder.add_user('Peter Kadletz', 'HZB', 3)
    builder.add_dataset(user_group, 'role', 'Beamline Responsible')
    user_group = builder.add_user('Robin Woracek', 'HZB', 4)
    builder.add_dataset(user_group, 'role', 'Beamline Responsible')
    user_group = builder.add_user('Nicklas Holmberg', 'ESS', 5)
    builder.add_dataset(user_group, 'role', 'Timing system')
    user_group = builder.add_user('Irina Stefanescu', 'ESS', 6)
    builder.add_dataset(user_group, 'role', 'DG contact')
    user_group = builder.add_user('Gregor Nowak', 'ESS', 7)
    builder.add_dataset(user_group, 'role', 'BEER detector team')
    user_group = builder.add_user('Michael Hart', 'STFC', 8)
    builder.add_dataset(user_group, 'role', 'V20 NICOS')
    user_group = builder.add_user('Matthew Jones', 'STFC', 9)
    builder.add_dataset(user_group, 'role', 'Streaming')
    user_group = builder.add_user('Owen Arnold', 'STFC', 10)
    builder.add_dataset(user_group, 'role', 'Mantid')
    user_group = builder.add_user('Neil Vaytet', 'ESS', 11)
    builder.add_dataset(user_group, 'role', 'Mantid Reduction, WFM treatment')
    user_group = builder.add_user('Torben Nielsen', 'ESS', 12)
    builder.add_dataset(user_group, 'role', 'Mantid/McStas')
    user_group = builder.add_user('Will Smith', 'STFC', 13)
    builder.add_dataset(user_group, 'role', 'Set-up of timing system')
    user_group = builder.add_user('Andrew Jackson', 'ESS', 14)
    builder.add_dataset(user_group, 'role', 'Observer')
    user_group = builder.add_user('Vendula Maulerova', 'ESS', 15)
    builder.add_dataset(user_group, 'role', 'Monitor tests')
    user_group = builder.add_user('Dominic Oram', 'STFC', 16)
    builder.add_dataset(user_group, 'role', 'ECP visitor')


def __add_monitors(builder):
    monitor_group_1 = builder.add_nx_group(instrument_group, 'monitor_1', 'NXmonitor')
    monitor_group_2 = builder.add_nx_group(instrument_group, 'monitor_2', 'NXmonitor')


if __name__ == '__main__':
    output_filename = 'V20_example_2.nxs'
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

        # Copy event data into detector
        __copy_existing_data()

        # TODO Add guides, shutters, anything else known

    with DetectorPlotter(output_filename, nx_entry_name) as plotter:
        plotter.plot_pixel_positions()
