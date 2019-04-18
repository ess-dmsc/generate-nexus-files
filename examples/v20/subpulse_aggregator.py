import h5py
import numpy as np
import argparse
from shutil import copyfile
from tqdm import tqdm
from matplotlib import pyplot as pl

# Uncomment for nicer styling on plots
# import seaborn as sns
# sns.set()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input-filename', type=str, help='Input file to convert.', required=True)
parser.add_argument('-o', '--output-filename', type=str, help='Output filename.')
parser.add_argument('--in-place', action='store_true',
                    help='Writes output data into the input file instead of creating a new output file')
parser.add_argument('--tdc-pulse-time-difference', type=int,
                    help='Time difference between TDC timestamps and pulse T0 in integer nanoseconds',
                    default=0)
parser.add_argument('--raw-event-path', type=str,
                    help='Path to the raw event NXevent_data group in the file',
                    default='/entry/instrument/detector_1/raw_event_data')
parser.add_argument('--output-event-path', type=str,
                    help='Path to the group where the output should be written',
                    default='/entry/event_data')
parser.add_argument('--chopper-tdc-path', type=str,
                    help='Path to the chopper TDC unix timestamps (ns) dataset in the file',
                    default='/entry/instrument/chopper_1/top_dead_centre_unix/time')
parser.add_argument('--wfm-chopper-tdc-path', type=str,
                    help='Path to the chopper TDC unix timestamps (ns) dataset in the file',
                    default='/entry/instrument/chopper_3/top_dead_centre_unix/time')
parser.add_argument('--wfm-2-chopper-tdc-path', type=str,
                    help='Path to the chopper TDC unix timestamps (ns) dataset in the file',
                    default='/entry/instrument/chopper_4/top_dead_centre_unix/time')
args = parser.parse_args()


def position_to_index(pos, count):
    uint_max = 2 ** 16 - 1
    # What if count (Nx or Ny) does not divide uint_max?
    return np.floor_divide(pos, (uint_max // count))


def convert_id(detector_id, id_offset=0):
    x = (detector_id[:] >> 16) & 0xffff
    y = detector_id[:] & 0xffff
    nx = 300
    ny = 300
    # Mantid requires 32 bit unsigned, so this should be correct dtype already.
    # Need offset here unless the banks event ids start at zero (Mantid
    # will simply discard events that do not correspond to IDF).
    return id_offset + position_to_index(x, nx) + nx * position_to_index(y, ny)


def write_event_data(output_data_group, event_ids, event_index_output, event_offset_output, event_time_zero_output):
    output_data_group.create_dataset('event_id', data=event_ids,
                                     compression='gzip',
                                     compression_opts=1)
    event_time_zero_ds = output_data_group.create_dataset('event_time_zero', data=event_time_zero_output,
                                                          compression='gzip',
                                                          compression_opts=1)
    event_time_zero_ds.attrs.create('units', np.array('ns').astype('|S2'))
    event_time_zero_ds.attrs.create('offset', np.array('1970-01-01T00:00:00').astype('|S19'))
    output_data_group.create_dataset('event_index', data=event_index_output.astype(np.uint64),
                                     compression='gzip',
                                     compression_opts=1)
    event_offset_ds = output_data_group.create_dataset('event_time_offset', data=event_offset_output,
                                                       compression='gzip',
                                                       compression_opts=1, dtype=np.uint32)
    event_offset_ds.attrs.create('units', np.array('ns').astype('|S2'))


def truncate_to_chopper_time_range(chopper_times, event_id_array, event_times, wfm_chopper_times, wfm_2_chopper_times):
    # Chopper timestamps are our pulse timestamps, we can only aggregate events per pulse
    # for time periods in which we actually have chopper timestamps
    # truncate any other events
    start = np.searchsorted(event_times, chopper_times[0], 'left')
    end = np.searchsorted(event_times, chopper_times[-2], 'right')
    event_id_array = event_id_array[start:end]
    event_times = event_times[start:end]

    # We need wfm chopper times up to the following chopper_time
    wfm_chopper_times = wfm_chopper_times[
                        np.searchsorted(wfm_chopper_times, chopper_times[0], 'left'):np.searchsorted(wfm_chopper_times,
                                                                                                     chopper_times[-1],
                                                                                                     'right')]
    wfm_2_chopper_times = wfm_2_chopper_times[
                          np.searchsorted(wfm_2_chopper_times, chopper_times[0], 'left'):np.searchsorted(
                              wfm_2_chopper_times,
                              chopper_times[-1],
                              'right')]

    chopper_times = chopper_times[:-2]

    return chopper_times, event_id_array, event_times, wfm_chopper_times, wfm_2_chopper_times


def _wfm_psc_1():
    """"
    Definition V20 for wfm pulse shaping chopper 1 (closest to source)
    :return: Returns the sorted angles of all edges in degrees. First entry is start angle of the first cut-out
    second entry is end angle of first cut-out. Cut-outs are in order from the position that the top-dead-centre (TDC)
    timestamp is recorded. The values in the array are from the closing edge of the largest window, TDC position is 15
    degrees after this.
    """
    return np.array([83.71, 94.7, 140.49, 155.79, 193.26, 212.56, 242.32, 265.33, 287.91, 314.37, 330.3, 360.0]) + 15.0


def _wfm_psc_2():
    """
    Definition V20 for wfm pulse shaping chopper 2 (closest to sample)
    :return: Returns the sorted angles of all edges in degrees. First entry is start angle of the first cut-out
    second entry is end angle of first cut-out. Cut-outs are in order from the position that the top-dead-centre (TDC)
    timestamp is recorded.
    """
    return np.array([65.04, 76.03, 126.1, 141.4, 182.88, 202.18, 235.67, 254.97, 284.73, 307.74, 330.00, 360.0]) + 15.0


def _tof_shifts(pscdata, psc_frequency=0.):
    """
    This is the time shift from the WFM chopper top-dead-centre timestamp to the t0 of each sub-pulse
    """
    cut_out_centre = np.reshape(pscdata, (len(pscdata) // 2, 2)).mean(1)
    tof_shifts = cut_out_centre / (360.0 * psc_frequency)
    # TODO What about the 17.1 degree phase shift from the chopper signal,
    #  which Peter mentioned, do we need to apply that here?
    return tof_shifts


def _which_subpulse(time_after_source_tdc):
    return np.searchsorted(threshold, time_after_source_tdc, 'left')


def plot_histograms(offset_from_source_chopper, offset_from_wfm_windows):
    fig, (ax1, ax2) = pl.subplots(2, 1)
    ax1.hist(offset_from_source_chopper, bins=2 * 144, range=(0, 72000000))
    for value in threshold:
        ax1.axvline(x=value, color='r', linestyle='dashed', linewidth=1)
    ax1.set_title(
        'Time offset from source chopper TDC timestamp, vertical lines indicate thresholds for subpulses')
    ax1.set_xlabel('Time (nanoseconds)')
    ax1.set_ylabel('Counts')
    ax2.hist(offset_from_wfm_windows, bins=2 * 144, range=(0, 72000000))
    ax2.set_title('Time offset from WFM chopper TDC timestamp, adjusted for the window timing for each subpulse')
    ax2.set_xlabel('Time (nanoseconds)')
    ax2.set_ylabel('Counts')
    # Leave a little more whitespace between the plots, so that the labels don't collide
    pl.subplots_adjust(hspace=0.3)


if __name__ == '__main__':
    # Nasty hardcoded thresholds for subpulses
    # TODO calculate these from beamline geometry
    component_name = args.raw_event_path.split('/')[-2]
    if component_name == "monitor_1":
        threshold = np.array([21300000, 31400000, 40500000, 48600000, 56600000], dtype=int)
    elif component_name == "monitor_2":
        threshold = np.array([20600000, 30600000, 39400000, 47600000, 55600000], dtype=int)
    else:
        threshold = np.array([21300000, 31500000, 40500000, 48500000, 56500000], dtype=int)
    low_cutoff = 10000000

    relative_shifts = (_tof_shifts(_wfm_psc_1(), psc_frequency=70.0) +
                       _tof_shifts(_wfm_psc_2(), psc_frequency=70.0)) * \
                      5.0e+08  # factor of 0.5 * 1.0e9 (taking mean and converting to nanoseconds)
    relative_shifts = relative_shifts.astype(np.uint64)

    if not args.in_place:
        print('Copying input file contents to output file...')
        copyfile(args.input_filename, args.output_filename)
        print('...done copying.')
        output_file = args.output_filename
    else:
        output_file = args.input_filename

    with h5py.File(output_file, 'r+') as raw_file:
        # Create output event group
        # output_data_group = raw_file['/entry'].create_group('event_data')
        output_data_group = raw_file.create_group(args.output_event_path)
        output_data_group.attrs.create('NX_class', 'NXevent_data', None, dtype='<S12')

        # Shift the TDC times
        source_tdc_times = raw_file[args.chopper_tdc_path][...]
        source_tdc_times += args.tdc_pulse_time_difference

        wfm_tdc_times = raw_file[args.wfm_chopper_tdc_path][...]
        wfm_2_tdc_times = raw_file[args.wfm_2_chopper_tdc_path][...]

        event_ids = raw_file[args.raw_event_path + '/event_id'][...]
        event_ids = convert_id(event_ids)

        event_time_zero_input = raw_file[args.raw_event_path + '/event_time_zero'][...]

        source_tdc_times, event_ids, event_time_zero_input, wfm_tdc_times, wfm_2_tdc_times = \
            truncate_to_chopper_time_range(source_tdc_times,
                                           event_ids,
                                           event_time_zero_input,
                                           wfm_tdc_times,
                                           wfm_2_tdc_times)

        source_tdc_index = 0
        wfm_tdc_index = 0
        wfm_2_tdc_index = 0  # for the second WFM chopper
        event_index = 0
        event_offset_output = np.zeros_like(event_time_zero_input, dtype=np.uint32)
        event_id_output = np.zeros_like(event_time_zero_input, dtype=np.uint32)
        offset_from_source_chopper_tdc = np.zeros_like(event_time_zero_input)
        event_index_output = np.array([0], dtype=np.uint64)
        event_time_zero_output = np.array([], dtype=np.uint64)
        subpulse_uuid = (0, 0)
        zero_time_events = 0
        print('Aggregating events by subpulse...')
        for event_input_number, (event_wallclock_time, event_id) in enumerate(
                tqdm(zip(event_time_zero_input, event_ids), total=len(event_time_zero_input))):
            if event_wallclock_time == 0:
                zero_time_events += 1
                continue
            # Find relevant source chopper timestamp
            tdc_index = np.searchsorted(source_tdc_times[source_tdc_index:], event_wallclock_time,
                                        'right') - 1 + source_tdc_index
            source_tdc = source_tdc_times[tdc_index]

            # Find relevant WFM chopper timestamps
            wfm_tdc_index = np.searchsorted(wfm_tdc_times[wfm_tdc_index:], source_tdc, 'right') + wfm_tdc_index
            wfm_tdc = wfm_tdc_times[wfm_tdc_index]
            wfm_2_tdc_index = np.searchsorted(wfm_2_tdc_times[wfm_2_tdc_index:], source_tdc, 'right') + wfm_2_tdc_index
            wfm_2_tdc = wfm_2_tdc_times[wfm_2_tdc_index]
            wfm_tdc_mean = np.uint64((wfm_tdc + wfm_2_tdc) // 2)

            # Determine which subpulse the event is in
            offset_from_source_chopper_tdc[event_input_number] = event_wallclock_time - source_tdc
            subpulse_index = _which_subpulse(offset_from_source_chopper_tdc[event_input_number])
            t0 = wfm_tdc_mean + relative_shifts[subpulse_index]

            if t0 > event_wallclock_time:
                # Throw this event away
                continue

            event_offset_output[event_index] = event_wallclock_time - t0
            event_id_output[event_index] = event_id

            next_subpulse_uuid = (wfm_tdc_index, subpulse_index)
            if next_subpulse_uuid == subpulse_uuid:
                # This event is from the same subpulse as the previous one
                event_index_output[-1] = event_index
            else:
                # Append a new subpulse
                # + 1 to event_index as it indicates the start of the next pulse, not end of current one
                event_index_output = np.concatenate((event_index_output, [event_index + 1]))
                event_time_zero_output = np.concatenate((event_time_zero_output, np.array([t0]).astype(np.uint64)))

            subpulse_uuid = next_subpulse_uuid
            event_index += 1

        # Truncate space from arrays which wasn't needed due to bad events
        event_offset_output = event_offset_output[:event_index]
        event_id_output = event_id_output[:event_index]
        offset_from_source_chopper_tdc = offset_from_source_chopper_tdc[:event_index]

        # Truncate last value as indicate start of a subpulse for which there were no events
        event_index_output = event_index_output[:-1]

        plot_histograms(offset_from_source_chopper_tdc, event_offset_output)

        write_event_data(output_data_group, event_id_output, event_index_output, event_offset_output,
                         event_time_zero_output)

        print(
            'WARNING, encountered {} event_time_zero entries with a value of 0, these were omitted from the output'.format(
                zero_time_events))

        pl.show()