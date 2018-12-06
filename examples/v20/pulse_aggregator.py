import h5py
import numpy as np


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
    # Shift the TDC times
    tdc_times = raw_file[chopper_tdc_path][...]
    tdc_times += tdc_pulse_time_difference_ns
    
