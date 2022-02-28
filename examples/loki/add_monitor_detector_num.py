from os import path

import numpy as np
import h5py

if __name__ == '__main__':
    file_name = '2021-02-23_0939.nxs'  # Just change file name here.
    file_path = path.join(path.dirname(__file__), file_name)

    fh = h5py.File(file_path, 'a')
    print(fh['entry']['instrument']['monitor_1'])
    fh['entry']['instrument']['monitor_1']['detector_id'] = np.array(1)
    fh['entry']['instrument']['monitor_2']['detector_id'] = np.array(2)
    fh.close()

