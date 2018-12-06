import h5py
import numpy as np

# Account for difference between TDC timestamps and the actual pulse T0
# due to angular separation fo the TDC position and the window on the chopper disk
tdc_pulse_time_difference = 0.0
raw_event_path = '/entry/instrument/detector_1/raw_event_data'

