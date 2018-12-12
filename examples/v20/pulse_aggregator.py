import h5py
import numpy as np
import argparse
from shutil import copyfile

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
                    default='/entry/instrument/chopper_1/top_dead_centre_unix/time')
args = parser.parse_args()


def position_to_index(pos, count):
    uint_max = 2 ** 16 - 1
    # What if count (Nx or Ny) does not divide uint_max?
    return np.floor_divide(pos, (uint_max // count))


def convert_id(event_id, id_offset=0):
    # TODO Is the order correct? Is x in the high bits or the low bits?
    x = (event_id[:] >> 16) & 0xffff
    y = event_id[:] & 0xffff
    Nx = 150
    Ny = 150
    # Mantid requires 32 bit unsigned, so this should be correct dtype already.
    # Need offset here unless the banks event ids start at zero (Mantid
    # will simply discard events that do not correspond to IDF).
    return id_offset + position_to_index(x, Nx) + Nx * position_to_index(y, Ny)


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
    with h5py.File(args.output_filename, 'r+') as raw_file:
        # Create output event group
        output_data_group = raw_file['/entry'].create_group('event_data')
        output_data_group.attrs.create('NX_class', 'NXevent_data', None, dtype='<S12')

        # Shift the TDC times
        tdc_times = raw_file[args.chopper_tdc_path][...]
        tdc_times += args.tdc_pulse_time_difference

        event_ids = raw_file[args.raw_event_path + '/event_id'][...]
        event_ids = convert_id(event_ids)

        event_time_zero_input = raw_file[args.raw_event_path + '/event_time_zero'][...]

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

        write_event_data(output_data_group, event_ids, event_index_output, event_offset_output, tdc_times)
