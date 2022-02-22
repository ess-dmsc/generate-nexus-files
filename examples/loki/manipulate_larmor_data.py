import numpy as np
import h5py

from LOKI_geometry import NexusFileLoader
from os import path


def get_nexus_np_array(data):
    arr = np.zeros(data.shape, dtype='float')
    data.read_direct(arr)
    return arr


if __name__ == '__main__':
    file_path = path.join(path.dirname(__file__), '2021-02-23_0939.nxs')
    nexus_loader = NexusFileLoader(file_path)
    nexus_loader.load_file()

    vertices_h5 = nexus_loader.get_data(
        f'entry.instrument.larmor_detector.pixel_shape.vertices')
    vertices = get_nexus_np_array(vertices_h5)
    vertices_transposed = vertices.T
    nexus_loader.close()
    fh = h5py.File(file_path, 'a')
    print(fh['entry']['instrument']['larmor_detector']
          ['pixel_shape']['vertices'][:, :])
    res = fh['entry']['instrument']['larmor_detector']['pixel_shape']['vertices']
    for i in range(3):
        for j in range(3):
            res[i, j] = vertices_transposed[i, j]
    print("________________________________________")
    print(fh['entry']['instrument']['larmor_detector']
          ['pixel_shape']['vertices'][:, :])

    try:
        res = fh['entry']['instrument']['monitor_1']['monitor_0_events']
        del res.attrs['topic']
        del res.attrs['source']
        res = fh['entry']['instrument']['monitor_2']['monitor_1_events']
        del res.attrs['topic']
        del res.attrs['source']
        res = fh['entry']['instrument']['larmor_detector']['larmor_detector_events']
        del res.attrs['topic']
        del res.attrs['source']
    except Exception as e:
        print('Error:', e)
    fh.close()






