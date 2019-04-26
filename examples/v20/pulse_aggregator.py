import h5py
import numpy as np
import argparse
from shutil import copyfile
import matplotlib.pylab as pl

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-filename", type=str, help='Input file to convert.')
parser.add_argument("-o", "--output-filename", type=str, help='Output filename.')
parser.add_argument("-t", "--tdc-pulse-time-difference", type=int,
                    help='Time difference between TDC timestamps and pulse T0 in integer nanoseconds',
                    default=0)
parser.add_argument("-e", "--raw-event-path", type=str,
                    help='Path to the raw event NXevent_data group in the file',
                    default='/entry/instrument/detector_1/raw_event_data')
parser.add_argument("-c", "--chopper-tdc-path", type=str,
                    help='Path to the chopper TDC unix timestamps (ns) dataset in the file',
                    default='/entry/instrument/chopper_1/top_dead_center/time')
args = parser.parse_args()


def position_to_index(pos, count):
    uint_max = 2 ** 16 - 1
    # What if count (Nx or Ny) does not divide uint_max?
    return np.floor_divide(pos, (uint_max // count))


def convert_id(event_id, id_offset=0):
    Nx = 512

    x = np.bitwise_and(event_id[:], 0xffff)
    y = np.right_shift(event_id[:], 16)

    # Hist, XEdge, YEdge = np.histogram2d(x, y, bins=(100, 100))
    # fig = pl.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(Hist)
    # pl.show()

    # Mantid requires 32 bit unsigned, so this should be correct dtype already.
    # Need offset here unless the banks event ids start at zero (Mantid
    # will simply discard events that do not correspond to IDF).
    # return id_offset + position_to_index(x, Nx) + Nx * position_to_index(y, Ny)
    return id_offset + x + (Nx * y)


def write_event_data(output_data_group, event_ids, event_index_output, event_offset_output, tdc_times):
    event_id_ds = output_data_group.create_dataset('event_id', data=event_ids,
                                                   compression='gzip',
                                                   compression_opts=1)
    event_time_zero_ds = output_data_group.create_dataset('event_time_zero', data=tdc_times,
                                                          compression='gzip',
                                                          compression_opts=1)
    event_time_zero_ds.attrs.create('units', np.array('ns').astype('|S2'))
    event_time_zero_ds.attrs.create('offset', np.array('1970-01-01T00:00:00').astype('|S19'))
    event_index_ds = output_data_group.create_dataset('event_index', data=event_index_output.astype(np.uint64),
                                                      compression='gzip',
                                                      compression_opts=1)
    event_offset_ds = output_data_group.create_dataset('event_time_offset', data=event_offset_output,
                                                       compression='gzip',
                                                       compression_opts=1)
    event_offset_ds.attrs.create('units', np.array('ns').astype('|S2'))


def truncate_to_chopper_time_range(chopper_times, event_id, event_times):
    # Chopper timestamps are our pulse timestamps, we can only aggregate events per pulse
    # for time periods in which we actually have chopper timestamps
    # truncate any other events
    start = np.searchsorted(event_times, chopper_times[0], 'left')
    end = np.searchsorted(event_times, chopper_times[-1], 'right')
    event_id = event_id[start:end]
    event_times = event_times[start:end]

    return chopper_times, event_id, event_times


if __name__ == '__main__':
    copyfile(args.input_filename, args.output_filename)
    with h5py.File(args.output_filename, 'r+') as output_file:
        # Create output event group
        output_data_group = output_file['/entry'].create_group('event_data')
        output_data_group.attrs.create('NX_class', 'NXevent_data', None, dtype='<S12')

        # Shift the TDC times
        tdc_times = output_file[args.chopper_tdc_path][...]
        tdc_times += args.tdc_pulse_time_difference

        event_ids = output_file[args.raw_event_path + '/event_id'][...]
        event_ids = convert_id(event_ids)
        print(np.max(event_ids))

        event_time_zero_input = output_file[args.raw_event_path + '/event_time_zero'][...]

        tdc_times, event_ids, event_time_zero_input = truncate_to_chopper_time_range(tdc_times, event_ids,
                                                                                     event_time_zero_input)

        event_index_output = np.zeros_like(tdc_times, dtype=np.uint64)
        event_offset_output = np.zeros_like(event_ids, dtype=np.uint32)
        event_index = 0
        for i, t in enumerate(tdc_times[:-1]):
            while event_index < len(event_time_zero_input) and event_time_zero_input[event_index] < tdc_times[i + 1]:
                # append event to pulse i
                if event_time_zero_input[event_index] > tdc_times[i]:
                    event_offset_output[event_index] = event_time_zero_input[event_index] - tdc_times[i]
                else:
                    raise Exception('Found event outside of chopper timestamp range, '
                                    'something went wrong when truncating the datasets')
                event_index += 1
            event_index_output[i + 1] = event_index

        event_ids[event_ids > 262143] = 262143
        write_event_data(output_data_group, event_ids, event_index_output, event_offset_output, tdc_times)

        # Delete the raw event data group
        del output_file[args.raw_event_path]

        # Delete waveform groups (not read by Mantid)
        for channel in range(3):
            group_name = f'/entry/instrument/detector_1/waveforms_channel_{channel}'
            del output_file[group_name]

        groups_to_remove = []


        def remove_groups_without_nxclass(name, object):
            if isinstance(object, h5py.Group):
                if 'NX_class' not in object.attrs.keys():
                    groups_to_remove.append(name)


        output_file.visititems(remove_groups_without_nxclass)

        for group in reversed(groups_to_remove):
            print(group)
            del output_file[group]

        pixels_per_axis = 512
        pixel_ids = np.arange(0, pixels_per_axis ** 2, 1, dtype=int)
        pixel_ids = np.reshape(pixel_ids, (pixels_per_axis, pixels_per_axis))
        del output_file['entry/instrument/detector_1/detector_number']
        output_file['entry/instrument/detector_1/'].create_dataset('detector_number', pixel_ids.shape, dtype=np.int64,
                                                                   data=pixel_ids)

        neutron_sensitive_width = 0.28  # metres, from DENEX data sheet
        # This pixel size is approximate, in practice the EFU configuration/calibration affects both the division
        # into 512 pixels and the actual active width we see of the detector
        # I suspect the actually detector area we collect data from is smaller than 0.28x0.28
        pixel_size = neutron_sensitive_width / pixels_per_axis
        single_axis_offsets = (pixel_size * np.arange(0, pixels_per_axis, 1,
                                                      dtype=np.float)) - (neutron_sensitive_width / 2.) + (
                                      pixel_size / 2.)
        x_offsets, y_offsets = np.meshgrid(single_axis_offsets,
                                           single_axis_offsets)

        del output_file['entry/instrument/detector_1/x_pixel_offset']
        del output_file['entry/instrument/detector_1/y_pixel_offset']
        xoffset_dataset = output_file['entry/instrument/detector_1/'].create_dataset('x_pixel_offset', x_offsets.shape,
                                                                                     dtype=np.float64, data=x_offsets)
        yoffset_dataset = output_file['entry/instrument/detector_1/'].create_dataset('y_pixel_offset', y_offsets.shape,
                                                                                     dtype=np.float64, data=y_offsets)
        del output_file['entry/monitor_1/waveforms']
        del output_file['entry/instrument/detector_1/waveforms_channel_3']

        # Patch the geometry ########################
        del output_file['entry/instrument/linear_axis_1']
        del output_file['entry/instrument/linear_axis_2']
        del output_file['entry/sample/transformations/offset_stage_1_to_default_sample']
        del output_file['entry/sample/transformations/offset_stage_2_to_sample']
        del output_file['entry/sample/transformations/offset_stage_2_to_stage_1']
        # Correct the source position
        output_file['entry/instrument/source/transformations/location'][...] = 27.4
        # Correct detector_1 position and orientation
        del output_file['entry/instrument/detector_1/depends_on']
        depend_on_path = '/entry/instrument/detector_1/transformations/x_offset'
        output_file['entry/instrument/detector_1'].create_dataset('depends_on', data=np.array(depend_on_path).astype(
            '|S' + str(len(depend_on_path))))
        del output_file['entry/instrument/detector_1/transformations/orientation']
        location_path = 'entry/instrument/detector_1/transformations/location'
        output_file[location_path][...] = 3.5
        output_file[location_path].attrs['vector'] = [0., 0., 1.]
        output_file[location_path].attrs['depends_on'] = '.'
        del output_file['entry/instrument/detector_1/transformations/beam_direction_offset']
        x_offset_dataset = output_file['entry/instrument/detector_1/transformations'].create_dataset('x_offset', (1,),
                                                                                                     dtype=np.float64,
                                                                                                     data=0.065)

        x_offset_dataset.attrs.create('units', np.array("m").astype('|S1'))
        translation_label = "translation"
        x_offset_dataset.attrs.create('transformation_type',
                                      np.array(translation_label).astype('|S' + str(len(translation_label))))
        x_offset_dataset.attrs.create('depends_on',
                                      np.array('/' + location_path).astype('|S' + str(len(location_path) + 1)))
        x_offset_dataset.attrs.create('vector',
                                      [-1., 0., 0.])
