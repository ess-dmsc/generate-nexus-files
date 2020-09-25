import numpy as np
from datetime import datetime


def add_example_nxlog(builder, parent_path='/raw_data_1/sample/', number_of_cues=1000):
    """
    Adds example NXlog class to the file
    """
    time = 0.0
    index = 0
    cue_timestamps = []
    cue_indices = []
    times = np.array([])
    values = np.array([])
    for cue_number in range(number_of_cues):
        number_of_samples = np.random.randint(number_of_cues * 10, number_of_cues * 20)
        cue_timestamps.append(time)
        cue_indices.append(index)
        time += 0.2 * number_of_cues + (np.random.rand() * 20)
        if cue_number > 0:
            values = np.hstack([values, np.sort(np.random.rand(number_of_samples) * (1/number_of_cues)) + values[-1]])
            times = np.hstack(
                (
                    times,
                    cue_timestamps[-1] + (np.sort(np.random.rand(number_of_samples)) * (time - cue_timestamps[-1]))))
        else:
            values = np.sort(np.random.rand(number_of_samples) * (1/number_of_cues)) + 0.21
            times = np.sort(np.random.rand(number_of_samples)) * time
        index += number_of_samples

    # Create an NXlog group in the sample group
    iso_timestamp = datetime.now().isoformat()
    data_group = builder.add_nx_group(parent_path, 'auxanometer_1', 'NXlog')
    builder.add_dataset(data_group, 'time', times.astype('float32'), {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(data_group, 'value', values.astype('float32'), {'units': 'cubits'})
    builder.add_dataset(data_group, 'cue_timestamp_zero', np.array(cue_timestamps).astype('float32'),
                        {'units': 's', 'start': iso_timestamp})
    builder.add_dataset(data_group, 'cue_index', np.array(cue_indices).astype('int32'))
