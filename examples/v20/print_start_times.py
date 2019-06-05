import argparse
import os
import h5py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input-directory', type=str,
                    help='Directory with .hdf files to print the start time of', required=True)
args = parser.parse_args()

filenames = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if
             os.path.isfile(os.path.join(args.input_directory, f))]

for filename in sorted(filenames):
    name, extension = os.path.splitext(filename)
    if extension == '.hdf':
        with h5py.File(filename, 'r+') as raw_file:
            start_time = str(raw_file['/entry/start_time'][...])
        print(f'{name.split("/")[-1]} start: {start_time}')
