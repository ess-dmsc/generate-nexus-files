import matplotlib.pyplot as plt
import numpy as np
from LOKI_geometry import NexusFileLoader
from os import path

def get_nexus_np_array(data):
    arr = np.zeros(data.shape, dtype='float')
    data.read_direct(arr)
    return arr


if __name__ == '__main__':
    file_path = path.join(path.dirname(__file__), 'loki.nxs')
    nexus_loader = NexusFileLoader(file_path)
    nexus_loader.load_file()
    detectors = []
    for i in range(0, 9):
        tmp_det = {'x_off': get_nexus_np_array(nexus_loader.get_data(
            f'entry.instrument.detector_{i}.x_pixel_offset')),
            'y_off': get_nexus_np_array(nexus_loader.get_data(
                f'entry.instrument.detector_{i}.y_pixel_offset')),
            'z_off': get_nexus_np_array(nexus_loader.get_data(
                f'entry.instrument.detector_{i}.z_pixel_offset'))}
        attribute = nexus_loader.get_attributes(
            f'entry.instrument.detector_{i}.transformations.trans_{i + 1}')
        data = nexus_loader.get_data(
            f'entry.instrument.detector_{i}.transformations.trans_{i + 1}.value',
            False)
        tmp_det['xyz'] = np.array((tmp_det['x_off'],
                                   tmp_det['y_off'],
                                   tmp_det['z_off'])).T
        tmp_det['transform'] = get_nexus_np_array(data)
        tmp_det['transform_vector'] = attribute.get('vector')
        tmp_det['xyz'] += tmp_det['transform'] * tmp_det['transform_vector']
        detectors.append(tmp_det)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    step_sz = 150
    for detector in detectors:
        data = detector['xyz'].T
        ax.scatter(data[0][0::step_sz],
                   data[1][0::step_sz],
                   data[2][0::step_sz])
    plt.show()


