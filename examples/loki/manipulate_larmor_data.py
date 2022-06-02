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

# import numpy as np
# import h5py
#
# from os import path
#
#
# def get_nexus_np_array(data):
#     arr = np.zeros(data.shape, dtype='float')
#     data.read_direct(arr)
#     return arr
#
#
# if __name__ == '__main__':
#     file_path = path.join(path.dirname(__file__), 'freia.nxs')
#     fh = h5py.File(file_path, 'a')
#     entry = fh['entry-01']
#
#     # fix instrument.
#     new_group = entry.create_group('instrument')
#     new_group.attrs['NX_class'] = 'NXinstrument'
#     new_group['freia_events'] = entry['freia_events']
#     new_group['PLC_senv_data'] = entry['PLC_senv_data']
#     new_group['m2_RBV'] = entry['m2_RBV']
#     new_group['m3_RBV'] = entry['m3_RBV']
#
#     # fix sample.
#     new_group = entry.create_group('sample')
#     new_group.attrs['NX_class'] = 'NXsample'
#     new_group['temperature'] = entry['m1_RBV']
#     new_group['temperature'].attrs['topic'] = 'Ymir_sample'
#     new_group['temperature'].attrs['source'] = 'sample_temperature'
#
#
#     try:
#         del entry['m1_RBV']
#         del entry['m2_RBV']
#         del entry['m3_RBV']
#         del entry['m4_RBV']
#         del entry['freia_events']
#         del entry['PLC_senv_data']
#         # res = fh['entry']['instrument']['monitor_1']['monitor_0_events']
#         # del res.attrs['topic']
#         # del res.attrs['source']
#         # res = fh['entry']['instrument']['monitor_2']['monitor_1_events']
#         # del res.attrs['topic']
#         # del res.attrs['source']
#         # res = fh['entry']['instrument']['larmor_detector']['larmor_detector_events']
#         # del res.attrs['topic']
#         # del res.attrs['source']
#     except Exception as e:
#         print('Error:', e)
#     fh.close()




