import h5py
import argparse
import numpy as np
import attr


@attr.s
class DatasetDetails(object):
    name = attr.ib()
    full_path = attr.ib()
    parent_path = attr.ib()
    text = attr.ib()


def find_variable_length_string_datasets(name, object):
    if isinstance(object, h5py.Dataset):
        if object.dtype == np.object:
            text = str(object[...])
            datasets_to_convert.append(
                DatasetDetails(object.name.split('/')[-1], object.name, '/'.join(object.name.split('/')[:-1]),
                               text))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input-filename', type=str, help='File to convert', required=True)
args = parser.parse_args()

with h5py.File(args.input_filename, 'r+') as file_to_convert:
    datasets_to_convert = []
    file_to_convert.visititems(find_variable_length_string_datasets)

    for dataset in datasets_to_convert:
        del file_to_convert[dataset.full_path]
        file_to_convert[dataset.parent_path].create_dataset(dataset.name, data=np.array(dataset.text).astype(
            '|S' + str(len(dataset.text))))
