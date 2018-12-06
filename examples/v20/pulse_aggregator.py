import h5py
import numpy as np


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


# Account for difference between TDC timestamps and the actual pulse T0
# due to angular separation of the TDC position and the window on the chopper disk
# In integer nanoseconds
tdc_pulse_time_difference_ns = 0

# Path of the NXevent_data group containing events which are not aggregated by pulse (event_time_offset is zeros)
raw_event_path = '/entry/instrument/detector_1/raw_event_data'

# Path of the TDC timestamp dataset (ns from unix epoch)
chopper_tdc_path = '/entry/instrument/chopper_1/top_dead_centre_unix/time'

filename = 'V20_raw_data.nxs'

with h5py.File(filename, 'r+') as raw_file:
    # Create output event group
    output_data_group = raw_file['/entry'].create_group('event_data')
    output_data_group.attrs.create('NX_class', 'NXevent_data', None, dtype='<S12')

    # Shift the TDC times
    tdc_times = raw_file[chopper_tdc_path][...]
    tdc_times += tdc_pulse_time_difference_ns

    event_ids = raw_file[raw_event_path + '/event_id'][...]
    event_ids = convert_id(event_ids)
    event_id_ds = output_data_group.create_dataset('event_id', data=event_ids,
                                                   compression='gzip',
                                                   compression_opts=1)

    event_time_zero_ds = output_data_group.create_dataset('event_time_zero', data=tdc_times,
                                                          compression='gzip',
                                                          compression_opts=1)
    event_time_zero_ds.attrs.create('units', np.array('ns').astype('|S2'))
    event_time_zero_ds.attrs.create('offset', np.array('1970-01-01T00:00:00').astype('|S19'))
