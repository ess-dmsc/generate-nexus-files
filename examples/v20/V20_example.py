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


def __copy_and_convert_logs(builder, nx_entry_name):
    __copy_log(builder, 'raw_data_1/selog/Guide_Pressure/value_log',
               nx_entry_name + '/instrument/guide_1/pressure', 'NXguide')
    __copy_log(builder, 'raw_data_1/framelog/proton_charge',
               nx_entry_name + '/instrument/source/proton_charge')
    __copy_log(builder, 'raw_data_1/runlog/dae_beam_current',
               nx_entry_name + '/instrument/source/beam_current')
    __copy_log(builder, 'raw_data_1/selog/Det_Temp_FLB/value_log',
               nx_entry_name + '/instrument/thermocouple_1/value_log', 'NXsensor')
    builder.add_dataset('instrument/thermocouple_1', 'measurement', 'temperature')
    builder.add_dataset('instrument/thermocouple_1', 'name', 'front-detector,left,bottom')
    __copy_log(builder, 'raw_data_1/selog/Det_Temp_FRB/value_log',
               nx_entry_name + '/instrument/thermocouple_2/value_log', 'NXsensor')
    builder.add_dataset('instrument/thermocouple_2', 'measurement', 'temperature')
    builder.add_dataset('instrument/thermocouple_2', 'name', 'front-detector,right,bottom')
    __copy_log(builder, 'raw_data_1/selog/Det_Temp_FLT/value_log',
               nx_entry_name + '/instrument/thermocouple_3/value_log', 'NXsensor')
    builder.add_dataset('instrument/thermocouple_3', 'measurement', 'temperature')
    builder.add_dataset('instrument/thermocouple_3', 'name', 'front-detector,left,top')
    __copy_log(builder, 'raw_data_1/selog/Det_Temp_FRT/value_log',
               nx_entry_name + '/instrument/thermocouple_4/value_log', 'NXsensor')
    builder.add_dataset('instrument/thermocouple_4', 'measurement', 'temperature')
    builder.add_dataset('instrument/thermocouple_4', 'name', 'front-detector,right,top')


if __name__ == '__main__':
    output_filename = 'V20_example.nxs'
    input_filename = 'adc_test8_half_cover_w_waveforms.nxs'  # None
    nx_entry_name = 'entry'
    # compress_type=32001 for BLOSC, or don't specify compress_type and opts to get non-compressed datasets
    with NexusBuilder(output_filename, input_nexus_filename=input_filename, nx_entry_name=nx_entry_name,
                      idf_file=None, compress_type='gzip', compress_opts=1) as builder:
        instrument_group = builder.add_instrument('V20', 'instrument')
        builder.add_user('Too many users to count', 'ESS and collaborators')

        # Add DENEX (delay line) detector geometry
        pixels_per_axis = 150  # 65535 (requires int64)
        detector_ids = np.reshape(np.arange(0, pixels_per_axis ** 2, 1, np.int32), (pixels_per_axis, pixels_per_axis))
        single_axis_offsets = (0.002 * np.arange(0, pixels_per_axis, 1, dtype=np.float)) - 0.15
        x_pixel_offsets, y_pixel_offsets = np.meshgrid(single_axis_offsets, single_axis_offsets)
        offsets = np.reshape(np.arange(0, pixels_per_axis ** 2, 1, np.int64), (pixels_per_axis, pixels_per_axis))
        builder.add_detector('DENEX delay line detector', 1, detector_ids,
                             {'x_pixel_offset': x_pixel_offsets, 'y_pixel_offset': y_pixel_offsets},
                             x_pixel_size=0.002, y_pixel_size=0.002)

        # TODO add two monitors and copy event data from adc_test8_half_cover_w_waveforms.nxs

        # Copy event data into detector
        # TODO once without downconversion of detector IDs to 150 by 150, once with conversion?
        __copy_existing_data()

        # TODO Add choppers - use diagram for positions, numbered from source end of beamline
        chopper_group_1 = builder.add_nx_group(instrument_group, 'chopper_1', 'NXdisk_chopper')
        builder.add_dataset(chopper_group_1, 'rotation_speed', 14.0, {'units': 'Hz'})
        tdc_log = builder.add_nx_group(chopper_group_1, 'top_dead_centre', 'NXlog')

        with h5py.File('chopper_tdc_file.hdf', 'r') as chopper_file:
            builder._NexusBuilder__copy_dataset(chopper_file['entry-01/ca_epics_double/time'], tdc_log.name + '/time')
            builder._NexusBuilder__copy_dataset(chopper_file['entry-01/ca_epics_double/value'], tdc_log.name + '/value')
            #tdc_log['time'].add

        # TODO Copy chopper TDCs from chopper_tdc_file.hdf into logs in the appropriate chopper
        # TODO Add guides, shutters, anything else known

        # __copy_and_convert_logs(builder, nx_entry_name)

    with DetectorPlotter(output_filename, nx_entry_name) as plotter:
        plotter.plot_pixel_positions()
