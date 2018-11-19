from collections import OrderedDict
from nexusutils.nexusbuilder import NexusBuilder
from nexusutils.detectorplotter import DetectorPlotter
import h5py
import numpy as np


def __copy_existing_data():
    """
    Copy data from the existing NeXus file to flesh out the example file
    """
    event_group_path = nx_entry_name + '/instrument/detector_1/event_data/'
    builder.copy_items(OrderedDict(
        [('raw_data_1/instrument/moderator', nx_entry_name + '/instrument/moderator'),
         ('raw_data_1/instrument/moderator/distance', nx_entry_name + '/instrument/moderator/distance'),
         ('raw_data_1/instrument/source/probe', nx_entry_name + '/instrument/source/probe'),
         ('raw_data_1/instrument/source/type', nx_entry_name + '/instrument/source/type'),
         ('raw_data_1/sample/name', nx_entry_name + '/sample/name'),
         ('raw_data_1/sample/type', nx_entry_name + '/sample/type'),
         ('raw_data_1/duration', nx_entry_name + '/duration'),
         ('raw_data_1/start_time', nx_entry_name + '/start_time'),
         ('raw_data_1/end_time', nx_entry_name + '/end_time'),
         ('raw_data_1/run_cycle', nx_entry_name + '/run_cycle'),
         ('raw_data_1/title', nx_entry_name + '/title'),
         ('raw_data_1/monitor_1/data', nx_entry_name + '/instrument/monitor_1/data'),
         ('raw_data_1/monitor_1/time_of_flight', nx_entry_name + '/instrument/monitor_1/time_of_flight'),
         ('raw_data_1/monitor_2/data', nx_entry_name + '/instrument/monitor_2/data'),
         ('raw_data_1/monitor_2/time_of_flight', nx_entry_name + '/instrument/monitor_2/time_of_flight'),
         ('raw_data_1/monitor_3/data', nx_entry_name + '/instrument/monitor_3/data'),
         ('raw_data_1/monitor_3/time_of_flight', nx_entry_name + '/instrument/monitor_3/time_of_flight'),
         ('raw_data_1/monitor_4/data', nx_entry_name + '/instrument/monitor_4/data'),
         ('raw_data_1/monitor_4/time_of_flight', nx_entry_name + '/instrument/monitor_4/time_of_flight'),
         ('raw_data_1/detector_1_events/', event_group_path),
         #('raw_data_1/detector_1_events/event_id', event_group_path + 'event_id'),
         ('raw_data_1/detector_1_events/event_index', event_group_path + 'event_index'),
         ('raw_data_1/detector_1_events/event_time_zero', event_group_path + 'event_time_zero'),
         ('raw_data_1/detector_1_events/event_time_offset', event_group_path + 'event_time_offset')
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

        builder.add_instrument('V20', 'instrument')
        builder.add_user('Too many users to count', 'ESS and collaborators')

        # Add DENEX (delay line) detector geometry
        pixels_per_axis = 150  # 65535 (requires int64)
        detector_ids = np.reshape(np.arange(0, pixels_per_axis**2, 1, np.int32), (pixels_per_axis, pixels_per_axis))
        single_axis_offsets = (0.002 * np.arange(0, pixels_per_axis, 1, dtype=np.float)) - 0.15
        x_pixel_offsets, y_pixel_offsets = np.meshgrid(single_axis_offsets, single_axis_offsets)
        offsets = np.reshape(np.arange(0, pixels_per_axis**2, 1, np.int64), (pixels_per_axis, pixels_per_axis))
        builder.add_detector('DENEX_detector', 1, detector_ids,
                             {'x_pixel_offsets': x_pixel_offsets, 'y_pixel_offsets': y_pixel_offsets},
                             x_pixel_size=0.002, y_pixel_size=0.002)

        # TODO Copy event data into detector
        # once without downconversion of detector IDs to 150 by 150, once with conversion

        # TODO Add choppers
        # TODO Copy chopper TDCs into logs in the appropriate chopper
        # TODO Add guides

        #__copy_existing_data()
        #__copy_and_convert_logs(builder, nx_entry_name)

    with DetectorPlotter(output_filename, nx_entry_name) as plotter:
        plotter.plot_pixel_positions()
